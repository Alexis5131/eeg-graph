"""Génère reports/results_summary.md à partir des sorties du notebook 03 et de scripts/mm_mi_cohort.py.

Les tableaux et les chiffres sont lus dans reports/tables/*.csv et reports/*.json ; la conclusion et les limites
sont recopiées depuis les cellules « Conclusion » et « Limites » du notebook 03. À relancer après toute
nouvelle exécution du notebook 03 :

    uv run python scripts/make_results_summary.py
"""

from __future__ import annotations

import json

import nbformat as nbf
import numpy as np
import pandas as pd

from eeg_graph.config import PROJECT_ROOT

R = PROJECT_ROOT / "reports"
T = R / "tables"


def md_table(df: pd.DataFrame) -> str:
    """Tableau Markdown (GitHub) d'un DataFrame, sans dépendance supplémentaire ; NaN -> cellule vide."""

    def cell(v: object) -> str:
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(row: pd.Series) -> str:
    return f"{row.median_ratio:.3f} [{row.ratio_ci_low:.3f}–{row.ratio_ci_high:.3f}]"


def pfmt(p: float) -> str:
    return f"{p:.2g}"


def main() -> int:
    nb = nbf.read(PROJECT_ROOT / "notebooks" / "03_mm_vs_mi.ipynb", as_version=4)
    conclusion = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("# Conclusion"))
    conclusion = conclusion.replace("# Conclusion\n", "").strip()
    limits = next(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("# Limites"))
    limits = limits.replace("# Limites\n", "").strip()

    # Tableau A : plafond aléatoire
    sim = pd.read_csv(T / "mm_mi_similarity_summary.csv")
    sim = sim[sim.density.isna() | (sim.density.round(2) == 0.15)]
    rows = []
    for comp in ("MM vs MI", "MM vs REST", "MI vs REST"):
        cells = [comp]
        for band in ("alpha", "beta"):
            for measure in ("edges", "strength"):
                r = sim[(sim.band == band) & (sim.measure == measure) & (sim.comparison == comp)].iloc[0]
                cells.append(f"{ci(r)} · {r.n_ratio_below_1}/{r.n}")
        rows.append("| " + " | ".join(cells) + " |")
    table_a = "\n".join(
        ["| Comparaison | α arêtes | α strength 15 % | β arêtes | β strength 15 % |", "|---|---|---|---|---|", *rows]
    )
    pmax_a = sim.p_holm.max()

    # Tableau B : plafond par run
    run = pd.read_csv(T / "mm_mi_run_similarity_summary.csv")
    rows = []
    for comp, label in (
        ("MM vs MI (runs, distance appariée)", "distance appariée (6 paires, 5.33 runs)"),
        ("MM vs MI (runs, 9 paires)", "toutes les paires (9 paires, 3.89 runs)"),
    ):
        cells = [label]
        for band in ("alpha", "beta"):
            for measure in ("edges", "strength"):
                r = run[(run.band == band) & (run.measure == measure) & (run.comparison == comp)].iloc[0]
                cells.append(f"{ci(r)} · {r.n_ratio_below_1}/{r.n} · p={pfmt(r.p_holm)}")
        rows.append("| " + " | ".join(cells) + " |")
    table_b = "\n".join(
        [
            "| Paires « entre » MM–MI | α arêtes | α strength 15 % | β arêtes | β strength 15 % |",
            "|---|---|---|---|---|",
            *rows,
        ]
    )

    # Profil de distance
    prof = (
        pd.read_csv(T / "mm_mi_run_pair_profile.csv")
        .groupby(["band", "measure", "type", "distance"])
        .similarity.median()
    )
    prof_table = prof.unstack(["type", "distance"]).round(3)
    prof_table.columns = [f"{t} d={d}" for t, d in prof_table.columns]
    prof_md = prof_table.reset_index().pipe(md_table)

    # Graphes de groupe
    grp = pd.read_csv(T / "mm_mi_group_graph_summary.csv")
    grp["isolés"] = grp.isolated_nodes.fillna("").str.count(",") + (grp.isolated_nodes.fillna("") != "").astype(int)
    cols = {
        "band": "bande",
        "condition": "condition",
        "n_edges": "arêtes",
        "n_components": "composantes",
        "isolés": "nœuds isolés",
        "diameter_largest_component": "diamètre (pl. comp.)",
        "radius_largest_component": "rayon (pl. comp.)",
        "diameter_mst_union": "diamètre (∪ MST)",
        "radius_mst_union": "rayon (∪ MST)",
        "modularity": "modularité",
        "n_communities": "communautés",
        "nmi_lobes": "NMI lobes",
    }
    grp_md = grp[list(cols)].rename(columns=cols).round(3).pipe(md_table)

    # Tests globaux
    tests = pd.read_csv(T / "mm_mi_global_tests.csv")
    tests = tests[tests.density.round(2) == 0.15]
    names = {
        "global_efficiency_weighted": "efficacité",
        "clustering_weighted": "clustering",
        "modularity": "modularité",
    }
    tests_md = (
        tests.assign(metric=tests.metric.map(names), p_holm=tests.p_holm.map(pfmt), W=tests.W.astype(int))[
            ["band", "metric", "comparison", "median_a", "median_b", "W", "p_holm", "r_rb"]
        ]
        .rename(
            columns={
                "band": "bande",
                "metric": "métrique",
                "comparison": "comparaison",
                "median_a": "médiane 1",
                "median_b": "médiane 2",
                "p_holm": "p Holm",
                "r_rb": "r rang-bisériale",
            }
        )
        .round(4)
        .pipe(md_table)
    )

    hubs = pd.read_csv(T / "mm_mi_hubs.csv").pipe(md_table)
    res = json.loads((R / "mm_vs_mi_results.json").read_text())
    overlap = pd.DataFrame(res["hub_overlap"]).round(2).pipe(md_table)
    roi = (
        pd.read_csv(T / "mm_mi_roi_strength_tests.csv")
        .assign(p_holm=lambda d: d.p_holm.map(pfmt), W=lambda d: d.W.astype(int))[
            ["band", "comparison", "median_a", "median_b", "W", "p_holm", "r_rb"]
        ]
        .round(4)
        .pipe(md_table)
    )
    comm = (
        pd.read_csv(T / "mm_mi_communities.csv")[["band", "comparison", "nmi", "ari", "module_jaccard", "n_common"]]
        .round(3)
        .pipe(md_table)
    )
    conf = (
        pd.read_csv(T / "mm_mi_power_strength_confound.csv").assign(p=lambda d: d.p.map(pfmt)).round(4).pipe(md_table)
    )
    robust = pd.read_csv(T / "mm_mi_robustness.csv").rename(columns={"result": "résultat"}).pipe(md_table)
    frac = res["robustness_holds_fraction"]
    cohort = json.loads((R / "cohort_mm_mi.json").read_text())
    erd = json.loads((R / "validation_task_cohort.json").read_text())["erd"]

    def erd_row(key: str) -> str:
        a, b = erd[key]["a_roi_in_top"], erd[key]["b_roi_erd_negative"]
        cells = [
            key.replace("_lr_", " "),
            f"{a['n_roi_in_top']}/6 (p = {a['hypergeom_p']:.1e})",
            f"{b['mean_power_change_pct']:.1f} %",
            f"{b['n_subjects_negative']}/{b['n_subjects']}",
            f"{b['wilcoxon_p_less']:.1e}",
        ]
        return "| " + " | ".join(cells) + " |"

    erd_md = "\n".join(
        [
            "| condition, bande | ROI dans le top 10 | variation de puissance de la ROI"
            " | sujets avec ERD < 0 | Wilcoxon p |",
            "|---|---|---|---|---|",
        ]
        + [erd_row(k) for k in ("mm_lr_alpha", "mm_lr_beta", "mi_lr_alpha", "mi_lr_beta")]
    )
    cmm = {b: erd[f"c_mm_vs_mi_{b}"] for b in ("alpha", "beta")}

    md = f"""# Résultats — Bouger ou imaginer bouger : est-ce le même réseau cérébral ?

Synthèse pour le rapport et les slides. Tous les chiffres viennent des fichiers cités (générés par
`scripts/mm_mi_cohort.py` et les notebooks 02 et 03). Chemins relatifs à `reports/`.

## Conclusion (notebook 03)

{conclusion}

## 1. Ressemblance MM vs MI rapportée à un plafond intra-condition

Ratio par sujet = similarité entre conditions / similarité intra-condition (ρ de Spearman), à nombre d'epochs
égal. Cellule : médiane [IC 95 % bootstrap, 5 000 tirages sur les sujets] · sujets avec ratio < 1.

**Tableau A — plafond 1, moitiés aléatoires** (105 sujets, n/2 = 21 epochs par moitié en médiane ;
Wilcoxon entre vs intra : toutes les p de Holm ≤ {pmax_a:.1e}). Source : `tables/mm_mi_similarity_summary.csv`.

{table_a}

Phrase clé : MM et MI se ressemblent à 94 % (alpha) et 96 % (bêta) du plafond sur les arêtes, IC sous 1 ;
l'écart est du même ordre pour chaque tâche face au repos (0.90 à 0.97).

**Tableau B — plafond 2, par run** (104 sujets, S079 écarté ; 13 epochs tâche par run en médiane ; intra =
6 paires de runs d'une même condition, distance moyenne 5.33 runs).
Source : `tables/mm_mi_run_similarity_summary.csv`.

{table_b}

Phrase clé : à distance temporelle appariée, MM–MI reste sous le plafond (0.95 en alpha, 0.97 en bêta sur les
arêtes, IC sous 1) ; avec toutes les paires, plus proches dans le temps, le ratio approche 1
(bêta : IC contenant 1).

**Figure — profil de distance** : `figures/mm_mi_run_ceiling_distance.png`.

![Similarité selon la distance entre runs](figures/mm_mi_run_ceiling_distance.png)

Phrase clé : la ressemblance entre runs baisse avec leur distance (arêtes alpha, paires MM–MI : 0.359 à 1 run,
0.309 à 9 runs), et à distance égale une paire intra-condition reste plus semblable qu'une paire MM–MI
(arêtes alpha : 0.367 en intra à 4 runs contre 0.338 et 0.329 entre à 3 et 5 runs). Médianes des sujets
(`tables/mm_mi_run_pair_profile.csv`) :

{prof_md}

**Figure — boîtes entre / intra (plafond 1)** : `figures/mm_mi_similarity_boxes.png`. Les boîtes « intra »
dominent les boîtes « entre » pour les trois paires de conditions, les deux bandes et les deux mesures.

## 2. Limites

{limits}

## 3. Figures et tableaux pour le rapport

### Validation physiologique (ERD, 105 sujets)

Figures `figures/validation_task_erd_cohort_{{mm,mi}}_lr_{{alpha,beta}}.png` (barres des 20 régions les plus
désynchronisées, ROI en gras). Source : `validation_task_cohort.json`.

{erd_md}

Phrase clé : la chaîne retrouve la désynchronisation sensorimotrice attendue, plus forte en MM qu'en MI
(chez {cmm["alpha"]["n_subjects_mm_stronger"]}/105 sujets en alpha, p = {cmm["alpha"]["wilcoxon_p_less"]:.1e} ;
{cmm["beta"]["n_subjects_mm_stronger"]}/105 en bêta, p = {cmm["beta"]["wilcoxon_p_less"]:.1e}).

### Cohorte

`tables/cohort_mm_mi.csv`, `cohort_mm_mi.json` : {cohort["n_included"]} sujets inclus sur {cohort["n_subjects"]}
(S088, S092, S100 : runs à 128 Hz ; S091 : 15 epochs MM < 20).
Après égalisation : n = {cohort["n_equalized"]["50%"]:.0f}
epochs par condition en médiane ({cohort["n_equalized"]["min"]:.0f}-{cohort["n_equalized"]["max"]:.0f}).

### Graphes de groupe (wPLI, 15 % = 342 arêtes)

`tables/mm_mi_group_graph_summary.csv`. Les nœuds isolés comptent chacun comme une communauté.

{grp_md}

Phrase clé : les trois graphes ont une organisation très proche (diamètre 4-5, modularité 0.30-0.32) ; les
nœuds isolés sont surtout frontaux (medialorbitofrontal-lh, parstriangularis-lh, rostralmiddlefrontal-lh).

Figures : `figures/mm_mi_group_matrices_{{alpha,beta}}.png` (matrices MM / MI / REST côte à côte, même échelle) et
`figures/mm_mi_group_brain_{{MM,MI,REST}}_{{alpha,beta}}.png` (vues cérébrales colorées par communauté).
Top 10 par strength, betweenness et eigenvector : `tables/mm_mi_group_top10.csv` (paracentral-lh est 1er en
strength et en betweenness en alpha dans les trois conditions).

### Métriques globales par sujet (15 %, Holm sur 9 tests par bande)

`tables/mm_mi_global_tests.csv`, figure `figures/mm_mi_global_metrics_paired.png`.

{tests_md}

Phrase clé : aucune différence MM vs MI après correction ; en bêta, MM et MI sont moins modulaires que le repos
(r = −0.51 et −0.39) et MM moins efficace que le repos (r = −0.36).

### Hubs (7 régions de plus forte strength moyenne) et ROI sensorimotrice

`tables/mm_mi_hubs.csv` :

{hubs}

Recouvrement (Jaccard) :

{overlap}

Strength moyenne de la ROI (pre/post/paracentral, deux côtés) contre REST, `tables/mm_mi_roi_strength_tests.csv` :

{roi}

Phrase clé : hubs identiques en alpha entre MM et MI ; en bêta, la ROI gagne en strength autant en MM qu'en MI
par rapport au repos.

### Communautés

`tables/mm_mi_communities.csv` (partitions de Louvain des graphes de groupe ; module sensorimoteur = communauté
qui contient le plus de régions de la ROI) :

{comm}

Phrase clé : partitions MM et MI proches (NMI 0.70 en alpha, 0.93 en bêta) ; le module sensorimoteur commun
(15 régions en alpha, 16 en bêta) regroupe precentral, postcentral et paracentral droits avec des régions
pariéto-temporo-occipitales droites.

### Contrôle de confusion puissance / connectivité

`tables/mm_mi_power_strength_confound.csv` (Spearman sur les sujets entre Δ puissance et Δ strength de la ROI,
MM − MI) :

{conf}

Phrase clé : en alpha, les sujets dont la puissance baisse le plus en MM perdent aussi le plus de strength
(ρ = 0.29, p = 0.003) ; rapporté, pas corrigé.

### Robustesse (seuils 10 / 15 / 20 %, alpha et bêta)

`tables/mm_mi_robustness.csv` (définition clémente ; stricte : `tables/mm_mi_robustness_strict.csv`).
{frac["lenient"]:.0%} des conclusions tiennent (clémente), {frac["strict"]:.0%} (stricte).

{robust}

Phrase clé : la ressemblance « entre < intra » tient partout ; les seuls changements portent sur l'efficacité
globale et la modularité MI vs REST en bêta à 10 %.

### Validation de la chaîne sur le repos (notebook 01)

`figures/validation_V1_source.png` (effet Berger en source, 10 sujets) et `figures/validation_V6_sub-001.png`
(sensibilité à l'inverse). Chiffres : `validation_source.json`, `note_pipeline.md`.
    """
    out = R / "results_summary.md"
    out.write_text("\n".join(line.rstrip() for line in md.strip().split("\n")) + "\n")
    print(f"Écrit : {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
