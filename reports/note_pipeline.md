# Note — chaîne EEG → graphe (6/11/2026)

1. **Temps** : ~7 s par enregistrement de 1 min (ICA partagée comprise), 10 sujets × 2 runs en 42 s sur 4 workers ; forward + noyau eLORETA calculés une fois (17 s). Extrapolation : ~1 h sur 1 worker pour ~530 runs.
2. **Exclusions** : par run, sur contrôles mesurés. 88/92/100 : runs de tâche à 128 Hz (baselines OK) ; 104 R08 et 106 R05 tronqués (avertissement) ; baselines de 89 mal annotées (sans effet).
3. **Inverse** : eLORETA, λ² = 1, fsaverage, covariance ad hoc ; réduit à un noyau fixe W (68 × 64). Équivalence avec la chaîne MNE vérifiée (V5, tolérance relative 1e-6, eLORETA et dSPM).
4. **Robustesse (V6, S001)** : à λ² égal, eLORETA, dSPM et MNE concordent (ρ de Spearman 0.83 à 0.98). Changer λ² modifie le classement fin des arêtes (ρ 0.24 à 0.59). Le bloc postérieur EC reste stable (0.40 à 0.44). → reporter la sensibilité à λ² pour le use case retenu.
5. **Métriques** : wPLI, ciPLV, AEC orthogonalisée sur signal analytique (filtre FIR continu puis Hilbert), par epoch ; enregistrement = moyenne des epochs.
6. **Rejet d'epochs** : un seuil fixe de 150 µV rejetait 100 % des epochs de S001 (alpha EC à 400-500 µV crête-à-crête en occipital). Remplacé par un z-score robuste > 3 par enregistrement, pour éviter un biais de condition.
7. **Vérifié** : effet Berger en source (V1 ; S001, top = inferiorparietal-rh ; 10 sujets, top = cuneus-lh, 7/10 des régions de tête pariéto-occipitales) ; bloc wPLI alpha postérieur EC > EO (9/10 sujets, Wilcoxon p = 0.0098) ; validité des matrices (V2, 22 fichiers) ; 34 tests unitaires ; mode capteurs opérationnel.
8. **Non vérifié** : V3 (direction des effets de Fraschini sur 20 sujets) ; MST-union dans une analyse réelle ; runs d'imagerie (R04/R08/R12) jamais traités ; datasets sommeil / épilepsie ; dérivations bipolaires.
9. **Limites** : pas d'IRM individuelle (template fsaverage) ; plancher de bruit du wPLI sur 4 s ; ICA à 20 composantes avec Fp1/Fp2 comme proxy EOG.
10. **Suite** : choisir use case et dataset, puis `build_cache.py` sur la cohorte et analyse.
