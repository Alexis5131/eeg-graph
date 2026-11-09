# Bouger ou imaginer bouger : est-ce le même réseau cérébral ?

Projet Graph Analytics 2026 (Albert School, MSA-DAT07-05). À partir de l'EEG de 105 sujets, on construit un
graphe de connectivité cérébrale (68 régions corticales Desikan-Killiany en espace source, arêtes pondérées
par le wPLI, bandes alpha 8-13 Hz et bêta 13-30 Hz) pour trois conditions : mouvement réel du poing (MM),
imagination du même mouvement (MI) et pauses entre essais (REST). On se demande si MM et MI reposent sur le
même réseau.

## Résultats en bref

1. **Validation physiologique** : la puissance alpha/bêta baisse sur le cortex sensorimoteur pendant MM et MI
   (ERD, 105 sujets, p < 1e-15), plus fortement en MM (90 et 97 sujets sur 105, p < 1e-13).
2. **Même architecture, partagée avec le repos** : MM et MI ont les mêmes hubs (Jaccard des 7 hubs = 1.00 en
   alpha, 0.75 en bêta) et le même module sensorimoteur (Jaccard 0.75 et 0.94) ; les hubs sont en majorité ceux
   du repos (Jaccard 0.75). Aucune métrique globale ne diffère après correction de Holm (p ≥ 0.078), mais ce
   n'est pas une équivalence : l'efficacité globale est un peu plus faible en MM qu'en MI (d_z = −0.28 en alpha,
   IC 90 % [−0.45 ; −0.12] ; −0.21 en bêta, [−0.36 ; −0.05]).
3. **Ressemblance quasi maximale mais pas totale** : les arêtes MM–MI atteignent 94-96 % du plafond
   intra-condition (moitiés aléatoires) et 95-97 % d'un plafond par run à distance temporelle appariée ;
   IC 95 % bootstrap sous 1.
4. **Effets de tâche partagés** : en bêta, la strength de la ROI sensorimotrice augmente autant en MM qu'en MI
   par rapport au repos (r = 0.45 et 0.48, p de Holm ≤ 5e-5) ; rien de tel en alpha.
5. **Conclusion** : une même organisation de réseau, dominée par une structure stable commune aux trois
   conditions et faiblement mais reproductiblement modulée par la condition (94 % des conclusions tiennent aux
   seuils 10 / 20 %). Le wPLI par epoch de 3.5 s a un plancher non nul sous indépendance (0.22-0.28 en alpha,
   0.13-0.16 en bêta, `reports/wpli_null_floor.json`). Détails, figures et limites : `reports/results_summary.md`.

## Données

PhysioNet **EEG Motor Movement/Imagery Database** v1.0.0 (EEGMMIDB),
https://physionet.org/content/eegmmidb/1.0.0/, doi:10.13026/C28G6P, licence **Open Data Commons Attribution
v1.0** (Schalk et al. 2004 ; Goldberger et al. 2000). 109 sujets, 64 électrodes 10-10, 160 Hz, EDF+, 14 runs
par sujet. **Les données ne sont pas versionnées** : elles se téléchargent automatiquement au premier
lancement (`mne.datasets.eegbci`) dans `data/`, avec le template `fsaverage` (`mne.datasets.fetch_fsaverage`,
~760 Mo).

Runs utilisés :

| Usage | Runs | Sujets |
|---|---|---|
| Analyse MM / MI / REST | R03, R07, R11 (poing gauche / droit réel) ; R04, R08, R12 (imaginé) | 1 à 109, 105 inclus |
| Validation de la chaîne (notebook 01) | R01 (repos yeux ouverts), R02 (repos yeux fermés) | 1 à 10 |
| Validation ERD (notebook 02) | R03, R04, R07, R08, R11, R12 | 1 à 10 |

Contrôles par run (`loading.check_eegmmidb_run` : fréquence, canaux, durée, durée des essais), puis règle
d'inclusion (≥ 2 runs exploitables et ≥ 20 epochs tâche par condition, `reports/tables/cohort_mm_mi.csv`) :

| Sujet | Décision | Raison |
|---|---|---|
| S088, S092, S100 | exclus | runs de tâche à 128 Hz, essais de 5.125 s au lieu de 4.1 s (autre protocole) |
| S091 | exclu | 15 epochs tâche en MM après rejet crête-à-crête (< 20) |
| S079 | inclus, sauf plafond par run | runs R04 et R07 avec 0 et 1 epoch tâche (règle : ≥ 8 epochs par run) |
| S104 | inclus | R08 tronqué (106 s), avertissement |
| S089 | inclus | R03 de 181 s avec 22 essais au lieu de 15 ; fenêtres vérifiées |

## Reproduction

Prérequis : [uv](https://docs.astral.sh/uv/), Python 3.12, ~6.5 Go de disque pour la chaîne complète, 16 Go de
RAM pour 4 workers (~1.2 Go par worker). Temps mesurés sur un Mac 8 cœurs, 16 Go de RAM.

| # | Commande | Ce que fait l'étape | Temps | Disque |
|---|---|---|---|---|
| 0 | `uv sync` | environnement `.venv` (dépendances figées par `uv.lock`) | 1 s avec le cache uv, sinon le téléchargement des paquets | ~0.5 Go (`.venv`) |
| 1 | `uv run python scripts/build_cache.py --subjects 1-109 --runs 3,4,7,8,11,12 --n-jobs 4` | télécharge les EDF et fsaverage, contrôles, ICA, noyau eLORETA, connectivité par epoch → `data/cache/` ; logs dans `logs/` | 17.2 min (4 workers, téléchargements compris) | EDF 1.6 Go, fsaverage 0.76 Go, cache 3.5 Go |
| 2 | `uv run python scripts/mm_mi_cohort.py` | inclusion des sujets, ERD sur la cohorte, matrices wPLI égalisées par sujet et condition, matrices par run → `data/derived/` | 33 s | 41 Mo |
| 3 | `notebooks/01_eeg_to_graph.ipynb` | validation de la chaîne sur le repos yeux ouverts / fermés (effet Berger, V1-V6), graphe et métriques d'un sujet | 59 s (données présentes) ; au premier lancement, + téléchargement de R01/R02 et de fsaverage | ~60 Mo de cache |
| 4 | `notebooks/02_task_erd_validation.ipynb` | découpage des runs de tâche et contrôle ERD sur les sujets 1-10 | 17 s si le cache existe ; sinon il calcule les 10 sujets (~100 s sur 4 workers + téléchargements) | inclus dans l'étape 1 |
| 5 | `notebooks/03_mm_vs_mi.ipynb` | **analyse** : graphes de groupe, comparaison par sujet, robustesse, conclusion | 2-3 min (134 s mesurées ; 173 s sur un clone neuf) | < 5 Mo de sorties |
| 6 | `uv run python scripts/wpli_null_floor.py` | plancher du wPLI par epoch sous indépendance (simulation, même filtre, même durée et même nombre d'epochs que l'analyse) → `reports/wpli_null_floor.json` | 3 s | < 5 ko |
| 7 | `uv run python scripts/make_results_summary.py` | régénère `reports/results_summary.md` depuis les tableaux, JSON et cellules « Conclusion » / « Limites » du notebook 03 | < 5 s | < 50 ko |

Un notebook s'exécute sans interface avec :

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_mm_vs_mi.ipynb
```

**Les étapes 1 et 2 sont facultatives pour reproduire l'analyse** : les matrices dérivées
(`data/derived/matrices_mm_mi.npz` 25 Mo, `data/derived/run_matrices_mm_mi.npz` 16 Mo, et leurs CSV de
métadonnées) sont versionnées. Sur un clone neuf, `uv sync` puis le notebook 03 suffisent (vérifié : résultats
identiques) ; seul fsaverage (~760 Mo) est téléchargé, pour les vues cérébrales. Les notebooks 01 et 02
téléchargent et calculent eux-mêmes ce dont ils ont besoin.

Les sorties des notebooks sont retirées à chaque commit par le filtre git `nbstripout` (déclaré dans
`.gitattributes`) ; l'installer une fois par clone avec `uv run nbstripout --install`.

Contrôle qualité :

```bash
uv run pytest                                      # tests unitaires + équivalence du noyau (V5)
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
```

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| `src/eeg_graph/` | package : chargement, prétraitement, espace source, connectivité, cache, graphe, métriques, validation, `task_analysis` (analyse MM / MI) |
| `scripts/` | `build_cache.py` (cache de la cohorte), `mm_mi_cohort.py` (inclusion, ERD, matrices dérivées), `make_results_summary.py` (synthèse des résultats) |
| `notebooks/` | 01 validation repos, 02 validation ERD, 03 analyse MM vs MI |
| `tests/` | tests unitaires (pytest) |
| `data/derived/` | matrices dérivées versionnées (le reste de `data/` est ignoré) |
| `reports/` | `results_summary.md` (résultats pour le rapport), `note_pipeline.md`, JSON de validation et de résultats, `figures/`, `tables/` |

## Pipeline (détail technique)

| Étape | Module | Choix |
|---|---|---|
| Chargement | `loading.py` | `eegbci.load_data`, `standardize`, montage `fsaverage_1005` |
| Prétraitement | `preprocessing.py` | FIR zero-phase 0.5-45 Hz ; référence moyenne (projecteur) ; ICA picard 20 composantes par sujet (ajustée sur copie filtrée 1 Hz, runs concaténés) ; composantes oculaires par corrélation avec Fp1/Fp2 (pas de canal EOG), 3 max |
| Epochs | `preprocessing.py` | fenêtres fixes (repos : 4 s, ou 5 × 12 s pour le protocole de Fraschini) ou autour d'événements (runs de tâche poing gauche / droit, réel R03/R07/R11 et imaginé R04/R08/R12 : découpage `task_lr`, 0.5-4.0 s après T0 → `rest`, T1 → `left`, T2 → `right`) ; rejet par z-score robuste (> 3) de l'amplitude crête-à-crête max, plafond 1 mV |
| Nœuds source | `source.py` | forward fsaverage (BEM 3 couches, ico-5, `mindist=5`), covariance ad hoc, eLORETA, `loose=0.2`, `depth=0.8`, `pick_ori="normal"`, λ² = 1 ; 68 labels `aparc` ; extraction `mean_flip` |
| Connectivité | `connectivity.py` | par bande (δ 1-4, θ 4-8, α 8-13, β 13-30, γ 30-45 Hz) : filtre FIR sur le signal continu → Hilbert → par epoch : **wPLI**, **ciPLV**, **AEC orthogonalisée** ; matrice d'un enregistrement = moyenne des epochs |
| Cache | `cache.py` | un `.npz` par (sujet, run, découpage) : `conn (3, 5, n, n)`, `conn_epochs (n_epochs, 3, 5, n, n)`, `power_epochs`, attributs des nœuds, métadonnées ; `index.csv` (mis à jour sous verrou : les workers écrivent en parallèle) |
| Graphe | `graph.py` | seuillage proportionnel (10/15/20 % des paires) ou graphe pondéré complet ; union avec l'arbre couvrant maximal en option ; distance = 1/poids |
| Conditions MM / MI / REST | `task_analysis.py` | runs de tâche : MM = left + right de R03/R07/R11, MI = left + right de R04/R08/R12, REST = T0 des 6 runs ; par sujet, même nombre d'epochs n = min(n_MM, n_MI, n_REST) tiré au hasard (`SEED`, par sujet), moitiés split-half A / B de n // 2 epochs ; matrice = moyenne des epochs |
| Métriques | `metrics.py` | ordre, taille, densité, composantes, diamètre/rayon (sauts, sur la plus grande composante si non connexe), degré/strength, betweenness, closeness, eigenvector, clustering binaire et pondéré, efficacité globale binaire et pondérée, assortativité, Louvain (100 seeds, co-assignation, NMI/ARI vs lobes), spectre du Laplacien, connectivité algébrique, mesures de l'arbre couvrant |

**Noyau inverse fixe.** Template commun + covariance ad hoc + mêmes électrodes ⇒ l'opérateur inverse est le
même pour tous les enregistrements. L'inverse (`pick_ori="normal"`) et `mean_flip` étant linéaires,
l'ensemble se réduit à une matrice `W` (68 × 64) : `séries des régions = W @ X`. `W` s'obtient en appliquant
l'inverse à une matrice identité, comme le fait MNE dans `make_inverse_resolution_matrix`. L'équivalence
avec `apply_inverse_epochs` + `extract_label_time_course` est testée (`tests/test_source_kernel.py`).

## Validation de la chaîne

**Repos (notebook 01, R01/R02, 10 sujets)** : effet Berger retrouvé en source (puissance alpha EC/EO maximale
en occipital), bloc wPLI alpha pariéto-occipital plus fort yeux fermés (9/10 sujets, Wilcoxon p = 0.0098),
matrices valides, ~7 s par enregistrement. Voir `reports/note_pipeline.md`, `reports/validation_source.json`.

**ERD (notebook 02 et `scripts/mm_mi_cohort.py`)** : ERD par région = log10 P (epochs gauche + droite) −
log10 P (epochs T0), ROI = precentral, postcentral, paracentral des deux côtés. Sur 10 sujets : la ROI est
surreprésentée dans les 10 régions les plus négatives (5, 6, 3 et 4 régions sur 6 en MM alpha, MM bêta,
MI alpha, MI bêta ; p hypergéométrique ≤ 0.037), son ERD est négative (Wilcoxon unilatéral p ≤ 0.0049) et
plus forte en MM qu'en MI (p ≤ 0.002). Sur la cohorte (105 sujets) : 6, 6, 5 et 6 régions sur 6 dans le
top 10 ; ERD de la ROI négative chez 91 à 104 sujets (p < 1e-15) ; plus forte en MM chez 90 (alpha) et 97
(bêta) sujets (p < 1e-13). Détails : `reports/validation_task.json`, `reports/validation_task_cohort.json`.

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
- Statistiques du notebook 03 (`task_analysis.holm`, `task_analysis.rank_biserial`) : correction de Holm (1979)
  et corrélation rang-bisériale appariée (Kerby 2014), réimplémentées d'après les articles, pas de code repris.
- IC par bootstrap percentile sur les sujets (`task_analysis.bootstrap_median_ci`) : méthode classique
  (Efron & Tibshirani 1993), réimplémentée, pas de code repris.
- Formules des métriques citées dans `metrics.py` : clustering pondéré d'Onnela et al. (2005, implémentation
  NetworkX), efficacité globale pondérée (Latora & Marchiori 2001 ; Rubinov & Sporns 2010, implémentation
  maison), mesures de l'arbre couvrant (Stam et al. 2014).
- Plan d'analyse MM / MI (égalisation des epochs, plafonds split-half et par run, distance appariée) : défini
  par le groupe, sans source de code externe.
- Aucun code existant spécifique à EEGMMIDB n'a été repris.
- Bibliothèques : MNE-Python, mne-connectivity, python-picard, NumPy, SciPy, pandas, NetworkX,
  scikit-learn, matplotlib, plotly, joblib ; outils : uv, pytest, ruff, mypy, nbconvert, nbstripout.

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
- Efron B., Tibshirani R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Holm S. (1979). A simple sequentially rejective multiple test procedure. Scand J Stat 6(2):65-70.
- Kerby D. S. (2014). The simple difference formula: an approach to teaching nonparametric correlation. Compr Psychol 3:11.IT.3.1.
- Latora V., Marchiori M. (2001). Efficient behavior of small-world networks. Phys Rev Lett 87(19):198701.
- Onnela J.-P., Saramäki J., Kertész J., Kaski K. (2005). Intensity and coherence of motifs in weighted complex networks. Phys Rev E 71:065103.
- Pfurtscheller G., Lopes da Silva F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. Clin Neurophysiol 110(11):1842-1857.
- Rubinov M., Sporns O. (2010). Complex network measures of brain connectivity. NeuroImage 52(3):1059-1069.
- Stam C. J. et al. (2014). The trees and the forest: characterization of complex brain networks with minimum spanning trees. Int J Psychophysiol 92(3):129-138.
- Blondel V. D. et al. (2008). Louvain. J Stat Mech P10008.
