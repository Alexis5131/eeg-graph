# eeg-source-graph

Pipeline générique **EEG → graphe de connectivité cérébrale** pour le projet Graph Analytics 2026
(Albert School, MSA-DAT07-05). Un enregistrement EEG multicanal devient, pour chaque bande de fréquence
et chaque métrique de connectivité, un graphe pondéré non orienté simple dont les nœuds sont :

- **`source`** : les 68 régions corticales de l'atlas Desikan-Killiany (signaux reconstruits en espace
  source sur le template fsaverage) — exige un EEG dense (≥ 32 électrodes) en référence commune ;
- **`sensor`** : les électrodes (repli pour les EEG peu denses ou bipolaires).

Le use case (question posée aux graphes) n'est pas encore fixé ; EEGMMIDB yeux ouverts / yeux fermés
sert ici de **jeu de test** pour valider la chaîne.

## Données

PhysioNet **EEG Motor Movement/Imagery Database** v1.0.0 (EEGMMIDB) : 109 sujets, 64 électrodes 10-10,
160 Hz, EDF+, 14 runs par sujet (R01 repos yeux ouverts, R02 repos yeux fermés, R03-R14 tâches motrices).
Licence Open Data Commons Attribution v1.0. Les données ne sont pas versionnées : elles sont téléchargées
automatiquement depuis https://physionet.org/content/eegmmidb/1.0.0/ (via `mne.datasets.eegbci`) dans
`data/`, avec le template `fsaverage` (`mne.datasets.fetch_fsaverage`, ~760 Mo).

Contrôles par run (`loading.check_eegmmidb_run`), plutôt qu'une liste d'exclusion décrétée : fréquence
d'échantillonnage, nombre de canaux, durée, durée des essais. Constats sur les en-têtes des 1 526 EDF :
les runs de tâche de S088, S092 et S100 sont à 128 Hz avec un autre timing (exclus pour ces runs, leurs
baselines sont normales) ; S104 R08 et S106 R05 sont tronqués (avertissement) ; les baselines de S089 sont
mal annotées (sans effet : on découpe tout le run).

## Pipeline

| Étape | Module | Choix |
|---|---|---|
| Chargement | `loading.py` | `eegbci.load_data`, `standardize`, montage `fsaverage_1005` |
| Prétraitement | `preprocessing.py` | FIR zero-phase 0.5-45 Hz ; référence moyenne (projecteur) ; ICA picard 20 composantes par sujet (ajustée sur copie filtrée 1 Hz, runs concaténés) ; composantes oculaires par corrélation avec Fp1/Fp2 (pas de canal EOG), 3 max |
| Epochs | `preprocessing.py` | fenêtres fixes (repos : 4 s, ou 5 × 12 s pour le protocole de Fraschini) ou autour d'événements (runs de tâche poing gauche / droit, réel R03/R07/R11 et imaginé R04/R08/R12 : découpage `task_lr`, 0.5-4.0 s après T0 → `rest`, T1 → `left`, T2 → `right`) ; rejet par z-score robuste (> 3) de l'amplitude crête-à-crête max, plafond 1 mV |
| Nœuds source | `source.py` | forward fsaverage (BEM 3 couches, ico-5, `mindist=5`), covariance ad hoc, eLORETA, `loose=0.2`, `depth=0.8`, `pick_ori="normal"`, λ² = 1 ; 68 labels `aparc` ; extraction `mean_flip` |
| Connectivité | `connectivity.py` | par bande (δ 1-4, θ 4-8, α 8-13, β 13-30, γ 30-45 Hz) : filtre FIR sur le signal continu → Hilbert → par epoch : **wPLI**, **ciPLV**, **AEC orthogonalisée** ; matrice d'un enregistrement = moyenne des epochs |
| Cache | `cache.py` | un `.npz` par (sujet, run, découpage) : `conn (3, 5, n, n)`, `conn_epochs (n_epochs, 3, 5, n, n)`, `power_epochs`, attributs des nœuds, métadonnées ; `index.csv` |
| Graphe | `graph.py` | seuillage proportionnel (10/15/20 % des paires) ou graphe pondéré complet ; union avec l'arbre couvrant maximal en option ; distance = 1/poids |
| Métriques | `metrics.py` | ordre, taille, densité, composantes, diamètre/rayon (sauts, sur la plus grande composante si non connexe), degré/strength, betweenness, closeness, eigenvector, clustering binaire et pondéré, efficacité globale binaire et pondérée, assortativité, Louvain (100 seeds, co-assignation, NMI/ARI vs lobes), spectre du Laplacien, connectivité algébrique, mesures de l'arbre couvrant |

**Noyau inverse fixe.** Template commun + covariance ad hoc + mêmes électrodes ⇒ l'opérateur inverse est le
même pour tous les enregistrements. L'inverse (`pick_ori="normal"`) et `mean_flip` étant linéaires,
l'ensemble se réduit à une matrice `W` (68 × 64) : `séries des régions = W @ X`. `W` s'obtient en appliquant
l'inverse à une matrice identité, comme le fait MNE dans `make_inverse_resolution_matrix`. L'équivalence
avec `apply_inverse_epochs` + `extract_label_time_course` est testée (`tests/test_source_kernel.py`).

## Utilisation

**Toute l'exécution passe par le notebook [`notebooks/01_eeg_to_graph.ipynb`](notebooks/01_eeg_to_graph.ipynb)**
(environ 1 min une fois les données téléchargées). Le code est dans le package `eeg_graph` ; le notebook
enchaîne les étapes : chargement et contrôles, prétraitement et ICA, noyau inverse (V5), connectivité et
cache, graphe et métriques exigées, batch multi-sujets, validation (V1, V2, V4, V6), mode capteurs.

```bash
uv sync                                    # crée .venv avec les dépendances (dont ipykernel)
# ouvrir notebooks/01_eeg_to_graph.ipynb avec le noyau .venv, puis « Run All »
# ou sans interface :
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eeg_to_graph.ipynb
```

Les paramètres (sujet, runs, mode de nœuds, métrique, bande, seuils, sujets du batch) sont dans la
deuxième cellule de code.

Runs de tâche : [`notebooks/02_task_erd_validation.ipynb`](notebooks/02_task_erd_validation.ipynb) construit
le cache des runs R03/R04/R07/R08/R11/R12 (sujets 1 à 10, ~30 s par sujet sur 4 workers) et vérifie l'ERD
sensorimotrice ; il écrit `reports/validation_task.json` et `reports/figures/validation_task_erd_*.png`. Les sorties du notebook sont retirées automatiquement à chaque commit
(filtre git `nbstripout`, déclaré dans `.gitattributes`).

Contrôle qualité du code :

```bash
uv run pytest                                      # tests unitaires + équivalence du noyau (V5)
uv run ruff check . && uv run ruff format . && uv run mypy src tests
```

Sorties : `data/cache/eegmmidb/` (matrices, `index.csv`), `reports/figures/`, `reports/tables/`,
`reports/validation_source.json`, `reports/validation_task.json`.

## Validation (EEGMMIDB R01/R02)

Voir `reports/note_pipeline.md`. Résumé : effet Berger retrouvé en source (puissance alpha EC/EO maximale
en occipital, 10 sujets), bloc wPLI alpha pariéto-occipital plus fort yeux fermés (9/10 sujets, Wilcoxon
p = 0.0098), matrices valides, ~7 s par enregistrement.

## Validation (EEGMMIDB runs de tâche, contrôle ERD)

10 sujets, runs R03/R07/R11 (mouvement réel, MM) et R04/R08/R12 (mouvement imaginé, MI). ERD par région =
log10 P (epochs gauche + droite) − log10 P (epochs T0), ROI = precentral, postcentral, paracentral des deux
côtés. Détails et chiffres : `reports/validation_task.json`. En alpha et en bêta, MM et MI : la ROI est
surreprésentée dans les 10 régions les plus négatives (5, 6, 3 et 4 régions sur 6 ; p hypergéométrique ≤ 0.037),
l'ERD de la ROI est négative (Wilcoxon unilatéral p ≤ 0.0049) et plus forte en MM qu'en MI (p ≤ 0.002).

## Réutilisation de code

- Projet personnel précédent **Sleep-Analysis** (https://github.com/Alexis5131/Sleep-Analysis) :
  `reject_amplitude_artifacts` (`src/preprocessing.py`) adapté dans `preprocessing.epoch_ptp` ;
  `_save` (`src/visualization.py`) adapté dans `viz.save_figure`.
- Tutoriels et exemples MNE dont s'inspirent des portions du code :
  - forward EEG sur template : https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html
  - ICA : https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
  - connectivité de labels : https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_label_connectivity.html
  - principe de la matrice de résolution : https://mne.tools/stable/generated/mne.minimum_norm.make_inverse_resolution_matrix.html
- Définition de l'ERD (`validation.erd`, puissance pendant la tâche rapportée à une référence de repos) :
  Pfurtscheller G., Lopes da Silva F. H. (1999). Méthode reprise de l'article, pas de code.
- Aucun code existant spécifique à EEGMMIDB n'a été repris.
- Bibliothèques : MNE-Python, mne-connectivity, python-picard, NumPy, SciPy, pandas, NetworkX,
  scikit-learn, matplotlib.

## Déclaration d'usage de l'IA

Claude (Anthropic), via Claude Code, a été utilisé pour vérifier le cahier des charges (chiffres du
dataset, API MNE, références), planifier le pipeline et écrire une première version du code et des tests,
relus et validés par le groupe. *À compléter par le groupe avant le rendu.*

## Références

- Schalk G. et al. (2004). BCI2000: a general-purpose brain-computer interface system. IEEE TBME 51(6):1034-1043.
- Schalk G. (2009). EEG Motor Movement/Imagery Dataset v1.0.0. PhysioNet. doi:10.13026/C28G6P.
- Goldberger A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220.
- Gramfort A. et al. (2013). MEG and EEG data analysis with MNE-Python. Front Neurosci 7:267.
- Pascual-Marqui R. D. (2007). eLORETA. arXiv:0710.3341.
- Desikan R. S. et al. (2006). NeuroImage 31(3):968-980.
- Vinck M. et al. (2011). wPLI. NeuroImage 55(4):1548-1565.
- Bruña R., Maestú F., Pereda E. (2018). ciPLV. J Neural Eng 15(5):056011.
- Hipp J. F. et al. (2012). AEC orthogonalisée. Nat Neurosci 15(6):884-890.
- Nolan H., Whelan R., Reilly R. B. (2010). FASTER. J Neurosci Methods 192(1):152-162.
- Mahjoory K. et al. (2017). Consistency of EEG source localization and connectivity estimates. NeuroImage 152:590-601.
- Fraschini M., La Cava S. M., Didaci L., Barberini L. (2021). On the variability of functional connectivity and network measures in source-reconstructed EEG time-series. Entropy 23(1):5. arXiv:2007.05395.
- Pfurtscheller G., Lopes da Silva F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. Clin Neurophysiol 110(11):1842-1857.
- Rubinov M., Sporns O. (2010). Complex network measures of brain connectivity. NeuroImage 52(3):1059-1069.
- Blondel V. D. et al. (2008). Louvain. J Stat Mech P10008.
