"""
analyze_v3.py — Analyse de l'Expérience 3 : Effet des Personas
===============================================================
Usage :
    python analyze_v3.py
    python analyze_v3.py --exp3 results_exp3 --baseline results

Prérequis :
    pip install pandas matplotlib statsmodels tabulate scipy

Question de recherche :
    Le persona assigné (rationnel / prudent) modifie-t-il significativement
    le comportement stratégique des LLMs par rapport à la condition sans persona
    (exp 1, scénarios S1 et S3) ?

Sorties console :
    - KPIs par scénario (bluff rate, call rate, gains)
    - Tableau de comparaison exp3 vs baseline exp1 (delta + significativité)
    - Régression logistique : P(action) ~ C(carte) + C(persona)
    - Régression par modèle agrégée

Sorties fichiers (results_exp3/figures/) :
    - kpi_offense_table.png/pdf
    - kpi_defense_table.png/pdf
    - profit_curves.png/pdf
    - persona_comparison.png/pdf  → delta bluff/call par persona vs baseline
    - forest_persona.png/pdf      → OR des personas (rationnel vs sans persona)
    - forest_global.png/pdf
    - forest_sN.png/pdf
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from tabulate import tabulate

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

# Valeurs Nash théoriques
NASH = {
    "bluff_rate_x":  1 / 3,
    "value_bet_x":   1.0,
    "call_rate_j":   0.0,
    "call_rate_q":   1 / 3,
    "call_rate_k":   1.0,
}

NASH_COLOR = "#888888"
NASH_STYLE = {"color": NASH_COLOR, "linestyle": "--", "linewidth": 1.2, "alpha": 0.7}

# Couleurs personas
PERSONA_COLORS = {
    "rationnel":   "#1565C0",
    "prudent":     "#BF360C",
    "sans_persona": "#555555",
}

# Couleurs rôles forest plot
FP_COLOR_X  = "#1565C0"
FP_COLOR_Y  = "#BF360C"

# Mapping scénarios exp1 → baseline pour la comparaison
# exp1 S1 : Mistral(X) vs OpenAI(Y) sans historique
# exp1 S3 : OpenAI(X) vs Mistral(Y) sans historique
EXP1_BASELINE_SCENARIOS = {
    ("mistral-small-latest", "gpt-5.4-mini"):  1,   # S1 exp1
    ("gpt-5.4-mini", "mistral-small-latest"):  3,   # S3 exp1
    # self-play — pas de baseline directe dans exp1
}

# ==============================================================================
# 1. CLI
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_v3.py",
        description="Analyse exp3 — Effet des personas sur la rationalité stratégique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemples :
  python analyze_v3.py
  python analyze_v3.py --exp3 results_exp3 --baseline results
""",
    )
    p.add_argument(
        "--exp3",
        type=Path,
        default=Path("results_exp3"),
        metavar="DOSSIER_EXP3",
        help="Dossier des résultats exp3 (défaut : results_exp3/)",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("results"),
        metavar="DOSSIER_BASELINE",
        help="Dossier des résultats exp1 comme baseline (défaut : results/)",
    )
    return p

# ==============================================================================
# 2. CHARGEMENT
# ==============================================================================

def _load_dir(results_dir: Path, exp_label: str) -> dict[int, pd.DataFrame]:
    """Charge tous les scenario_N.csv d'un dossier."""
    dfs: dict[int, pd.DataFrame] = {}
    csv_files = sorted(results_dir.glob("scenario_*.csv"))

    if not csv_files:
        print(f"[WARN] Aucun fichier scenario_*.csv dans '{results_dir}'.")
        return dfs

    for path in csv_files:
        try:
            sid = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] {path.name} est vide — ignoré.")
            continue

        df["X_Action"] = pd.to_numeric(df["X_Action"], errors="coerce")
        df["Y_Action"] = pd.to_numeric(df["Y_Action"], errors="coerce")
        df["Fallback"] = df["Fallback"].astype(bool)
        df["Game_ID"]  = pd.to_numeric(df["Game_ID"],  errors="coerce")

        # Normalisation colonnes persona / historique
        if "Persona_X" not in df.columns:
            df["Persona_X"] = "sans_persona"
        if "Persona_Y" not in df.columns:
            df["Persona_Y"] = "sans_persona"
        if "History_Size" not in df.columns and "Has_History" in df.columns:
            df["History_Size"] = df["Has_History"].apply(lambda v: 20 if v else 0)
        elif "History_Size" not in df.columns:
            df["History_Size"] = 0

        n_fb = df["Fallback"].sum()
        print(f"  [{exp_label}] scenario_{sid}.csv : {len(df)} parties "
              f"({n_fb} fallbacks)")
        dfs[sid] = df

    return dfs


def load_data(exp3_dir: Path, baseline_dir: Path) -> tuple[
    dict[int, pd.DataFrame], dict[int, pd.DataFrame]
]:
    print("\n" + "=" * 70)
    print("  CHARGEMENT EXP3")
    print("=" * 70)
    dfs_exp3 = _load_dir(exp3_dir, "EXP3")

    print("\n" + "=" * 70)
    print("  CHARGEMENT BASELINE (EXP1)")
    print("=" * 70)
    dfs_base = _load_dir(baseline_dir, "BASE")

    # Filtrer baseline : garder uniquement S1 et S3 (sans historique)
    dfs_base = {
        sid: df for sid, df in dfs_base.items()
        if int(df["History_Size"].iloc[0]) == 0
    }
    print(f"  Scénarios baseline retenus (sans historique) : "
          f"{sorted(dfs_base.keys())}")

    return dfs_exp3, dfs_base


def _model_names(df: pd.DataFrame) -> tuple[str, str]:
    x = df["X_Model"].iloc[0].split("/")[-1] if "X_Model" in df.columns else "X"
    y = df["Y_Model"].iloc[0].split("/")[-1] if "Y_Model" in df.columns else "Y"
    return x, y


def _get_label(df: pd.DataFrame, sid: int) -> str:
    xm, ym = _model_names(df)
    px = df["Persona_X"].iloc[0]
    py = df["Persona_Y"].iloc[0]
    return f"S{sid} — {xm}[{px}](X) vs {ym}[{py}](Y)"


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = figures_dir / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"  [OK] {p.name}")
    plt.close(fig)

# ==============================================================================
# 3. KPIs
# ==============================================================================

def compute_kpis(df: pd.DataFrame) -> dict:
    clean = df[~df["Fallback"]]

    def bet_rate(sub: pd.DataFrame) -> float:
        return float((sub["X_Action"] == 2).mean()) if not sub.empty else float("nan")

    def call_rate(sub: pd.DataFrame, card: str) -> float:
        faced = sub[(sub["X_Action"] == 2) & (sub["Y_Card"] == card)]
        return float((faced["Y_Action"] == 4).mean()) if not faced.empty else float("nan")

    return {
        "n_total":      len(df),
        "n_clean":      len(clean),
        "n_fallback":   int(df["Fallback"].sum()),
        "bluff_rate_x": bet_rate(clean[clean["X_Card"] == "J"]),
        "value_bet_x":  bet_rate(clean[clean["X_Card"] == "K"]),
        "bet_rate_x":   bet_rate(clean),
        "call_rate_j":  call_rate(clean, "J"),
        "call_rate_q":  call_rate(clean, "Q"),
        "call_rate_k":  call_rate(clean, "K"),
        "x_win_rate":   float((clean["Winner"] == "X").mean()) if not clean.empty else float("nan"),
        "x_total_gain": float(clean["X_Net_Payoff"].sum()),
        "y_total_gain": float(clean["Y_Net_Payoff"].sum()),
    }


def _fmt(v: float, pct: bool = True) -> str:
    if np.isnan(v):
        return "N/A"
    return f"{v:.1%}" if pct else f"{v:+.0f}€"


def _nash_color(val: float, nash: float) -> str:
    if np.isnan(val):
        return "#555555"
    diff = abs(val - nash)
    if diff < 0.05:   return "#1B5E20"
    if diff < 0.20:   return "#E65100"
    return "#B71C1C"


def print_kpi_report(
    all_kpis: dict[int, dict],
    dfs: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Tableau KPI console — identique à analyze_v2."""
    rows_x = []
    for sid, k in sorted(all_kpis.items()):
        label = _get_label(dfs[sid], sid)
        rows_x.append([
            label, k["n_clean"], k["n_fallback"],
            _fmt(k["bluff_rate_x"]),
            _fmt(k["value_bet_x"]),
            _fmt(k["bet_rate_x"]),
            _fmt(k["x_win_rate"]),
            _fmt(k["x_total_gain"], False),
        ])

    print("\n" + "=" * 80)
    print("  JOUEUR X — Métriques offensives (Expérience 3)")
    print("=" * 80)
    print(tabulate(rows_x,
        headers=["Scénario", "N valides", "Fallbacks",
                 "Bluff(J)", "ValueBet(K)", "Bet%", "WinRate", "Gain net"],
        tablefmt="rounded_outline",
        colalign=("left","right","right","right","right","right","right","right"),
    ))
    print("  Nash théorique : Bluff(J) ≈ 33.3% | ValueBet(K) = 100%")

    rows_y = []
    for sid, k in sorted(all_kpis.items()):
        label = _get_label(dfs[sid], sid)
        rows_y.append([
            label, k["n_clean"],
            _fmt(k["call_rate_j"]),
            _fmt(k["call_rate_q"]),
            _fmt(k["call_rate_k"]),
            _fmt(k["y_total_gain"], False),
        ])

    print("\n" + "=" * 80)
    print("  JOUEUR Y — Métriques défensives (sur mains où X a Misé)")
    print("=" * 80)
    print(tabulate(rows_y,
        headers=["Scénario", "N valides", "Call(J)", "Call(Q)", "Call(K)", "Gain net"],
        tablefmt="rounded_outline",
        colalign=("left","right","right","right","right","right"),
    ))
    print("  Nash théorique : Call(Q) ≈ 33.3% | Call(K) = 100% | Call(J) = 0%")

    rows_export = []
    for sid, k in sorted(all_kpis.items()):
        rows_export.append({
            "Scenario_ID": sid,
            "Label": _get_label(dfs[sid], sid),
            **k,
        })
    return pd.DataFrame(rows_export)

# ==============================================================================
# 4. TABLEAU DE COMPARAISON EXP3 vs BASELINE EXP1
# ==============================================================================

def _find_baseline(df_exp3: pd.DataFrame,
                   dfs_base: dict[int, pd.DataFrame]) -> pd.DataFrame | None:
    """
    Trouve le scénario baseline exp1 correspondant à un scénario exp3.
    Critère : même X_Model, même Y_Model, History_Size == 0.
    """
    xm = df_exp3["X_Model"].iloc[0].split("/")[-1]
    ym = df_exp3["Y_Model"].iloc[0].split("/")[-1]

    for df_b in dfs_base.values():
        bxm = df_b["X_Model"].iloc[0].split("/")[-1]
        bym = df_b["Y_Model"].iloc[0].split("/")[-1]
        if bxm == xm and bym == ym:
            return df_b
    return None


def print_comparison_table(
    all_kpis_exp3: dict[int, dict],
    dfs_exp3: dict[int, pd.DataFrame],
    dfs_base: dict[int, pd.DataFrame],
) -> None:
    """
    Tableau de comparaison : exp3 persona vs exp1 baseline.
    Pour chaque scénario exp3, affiche le delta et teste la significativité
    via test z sur proportions (deux échantillons indépendants).
    """
    print("\n" + "=" * 90)
    print("  COMPARAISON EXP3 vs BASELINE EXP1 (sans persona)")
    print("  Test z sur proportions | * p<0.05 | ** p<0.01 | *** p<0.001")
    print("=" * 90)

    rows = []
    for sid, df_exp3 in sorted(dfs_exp3.items()):
        df_base = _find_baseline(df_exp3, dfs_base)
        if df_base is None:
            continue

        xm, ym       = _model_names(df_exp3)
        persona_x    = df_exp3["Persona_X"].iloc[0]
        persona_y    = df_exp3["Persona_Y"].iloc[0]
        clean3       = df_exp3[~df_exp3["Fallback"]]
        clean_b      = df_base[~df_base["Fallback"]]

        for metric_label, col_card, action_col, action_val, role in [
            ("Bluff(J)",  "X_Card", "X_Action", 2, "X"),
            ("ValueBet(K)", "X_Card", "X_Action", 2, "X"),
            ("Call(Q)",   "Y_Card", "Y_Action",  4, "Y"),
        ]:
            # Filtrage spécifique à la carte
            card = "J" if metric_label == "Bluff(J)" else \
                   "K" if metric_label == "ValueBet(K)" else "Q"

            if role == "X":
                sub3 = clean3[clean3[col_card] == card]
                sub_b = clean_b[clean_b[col_card] == card]
                n3, k3 = len(sub3), int((sub3[action_col] == action_val).sum())
                nb, kb = len(sub_b), int((sub_b[action_col] == action_val).sum())
            else:
                sub3  = clean3[(clean3["X_Action"] == 2) & (clean3[col_card] == card)]
                sub_b = clean_b[(clean_b["X_Action"] == 2) & (clean_b[col_card] == card)]
                n3, k3 = len(sub3), int((sub3[action_col] == action_val).sum())
                nb, kb = len(sub_b), int((sub_b[action_col] == action_val).sum())

            if n3 == 0 or nb == 0:
                continue

            p3  = k3 / n3
            pb  = kb / nb
            delta = p3 - pb

            # Test z sur deux proportions
            p_pool = (k3 + kb) / (n3 + nb)
            se     = np.sqrt(p_pool * (1 - p_pool) * (1/n3 + 1/nb))
            z      = (p3 - pb) / se if se > 0 else 0
            pval   = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
            sig    = "***" if pval < 0.001 else "**" if pval < 0.01 \
                     else "*" if pval < 0.05 else "ns"

            rows.append([
                f"S{sid}",
                f"{xm}[{persona_x}]",
                metric_label,
                f"{pb:.1%}",
                f"{p3:.1%}",
                f"{delta:+.1%}",
                f"{pval:.4f}",
                sig,
            ])

    print(tabulate(rows,
        headers=["Scén.", "Modèle X [Persona]", "Métrique",
                 "Baseline\n(exp1)", "Exp3\n(persona)",
                 "Δ", "p-value", "Sig."],
        tablefmt="rounded_outline",
        colalign=("center","left","left","right","right","right","right","center"),
    ))
    print("  Baseline = exp1 S1 (Mistral X) ou S3 (OpenAI X), sans persona.")

# ==============================================================================
# 5. RÉGRESSIONS LOGISTIQUES
# ==============================================================================

def _logit(sub: pd.DataFrame, formula: str) -> dict:
    """Régression logistique générique, retourne dict avec pvalues et OR."""
    try:
        model = smf.logit(formula, data=sub).fit(disp=0)
        res   = {"n": len(sub), "pseudo_r2": model.prsquared, "vars": []}
        for name, pval, coef in zip(
            model.pvalues.index,
            model.pvalues.values,
            model.params.values,
        ):
            sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                   else "*" if pval < 0.05 else "ns")
            res["vars"].append({
                "label": name,
                "pval":  float(pval),
                "coef":  float(coef),
                "or":    float(np.exp(coef)),
                "sig":   sig,
            })
        return res
    except Exception as e:
        return {"error": str(e)}


def _print_logit(res: dict, title: str) -> None:
    if "error" in res:
        print(f"    {title} : {res['error']}")
        return
    rows = []
    for v in res["vars"]:
        lbl = (v["label"]
               .replace("C(Card, Treatment('Q'))[T.", "Carte ")
               .replace("C(Persona, Treatment('sans_persona'))[T.", "Persona ")
               .replace("C(Persona, Treatment('rationnel'))[T.", "Persona ")
               .replace("C(Persona, Treatment('prudent'))[T.", "Persona ")
               .replace("]", "")
               .replace("Intercept", "Constante"))
        rows.append([lbl, f"{v['pval']:.4f}", v["sig"], f"{v['or']:.3f}"])
    print(f"\n    {title}  (N={res['n']}, pseudo-R²={res['pseudo_r2']:.3f})")
    print(tabulate(rows,
        headers=["Variable", "p-value", "Sig.", "Odds Ratio"],
        tablefmt="simple",
        colalign=("left", "right", "center", "right"),
    ))
    print("    *** p<0.001 | ** p<0.01 | * p<0.05 | ns = non significatif")


def run_regressions(
    dfs_exp3: dict[int, pd.DataFrame],
    dfs_base: dict[int, pd.DataFrame],
) -> dict:
    """
    Deux niveaux de régression :

    1. Par scénario exp3 : P(action) ~ C(carte, ref=Q)
       — même que analyze_v2 pour cohérence

    2. Régression persona — données exp3 + baseline poolées :
       P(action) ~ C(carte, ref=Q) + C(persona, ref='sans_persona')
       → le coefficient Persona mesure l'effet causal du persona
         après contrôle de la carte.
    """
    print("\n" + "=" * 80)
    print("  RÉGRESSIONS LOGISTIQUES — EXP3")
    print("  Référence carte = Q | Référence persona = sans_persona")
    print("=" * 80)

    collected: dict[int, dict] = {}

    # --- Par scénario exp3 (base) ---
    for sid, df in sorted(dfs_exp3.items()):
        clean = df[~df["Fallback"]].copy()
        label = _get_label(df, sid)
        xm, ym = _model_names(df)
        print(f"\n  ── {label} ──")

        sub_x = clean[["X_Card", "X_Action"]].rename(columns={"X_Card": "Card"})
        sub_x["Y_bin"] = (sub_x["X_Action"] == 2).astype(int)
        res_x = _logit(sub_x, "Y_bin ~ C(Card, Treatment('Q'))")
        _print_logit(res_x, f"X ({xm}) → P(Miser | Carte)")

        sub_y = (clean[clean["X_Action"] == 2][["Y_Card", "Y_Action"]]
                 .rename(columns={"Y_Card": "Card"})
                 .dropna(subset=["Y_Action"]))
        sub_y["Y_bin"] = (sub_y["Y_Action"] == 4).astype(int)
        res_y = _logit(sub_y, "Y_bin ~ C(Card, Treatment('Q'))")
        _print_logit(res_y, f"Y ({ym}) → P(Suivre | Carte)")

        collected[sid] = {
            "label": label, "xm": xm, "ym": ym,
            "res_x": res_x, "res_y": res_y,
        }

    # --- Régression persona poolée (exp3 + baseline) ---
    print("\n" + "=" * 80)
    print("  RÉGRESSION PERSONA — Données poolées exp3 + baseline")
    print("  P(action) ~ C(carte) + C(persona)  |  ref persona = sans_persona")
    print("=" * 80)

    # Pool exp3 + baseline pour X
    all_x_frames = []
    all_y_frames = []

    for df in dfs_base.values():
        c = df[~df["Fallback"]].copy()
        c["Persona"] = "sans_persona"
        all_x_frames.append(c[["X_Card", "X_Action", "Persona",
                                "X_Model"]].rename(columns={"X_Card": "Card",
                                                             "X_Action": "Action"}))
        fy = c[c["X_Action"] == 2][["Y_Card", "Y_Action", "Persona",
                                     "Y_Model"]].rename(
            columns={"Y_Card": "Card", "Y_Action": "Action",
                     "Y_Model": "Model"}).dropna(subset=["Action"])
        all_y_frames.append(fy)

    for df in dfs_exp3.values():
        c = df[~df["Fallback"]].copy()
        c["Persona"] = c["Persona_X"]
        all_x_frames.append(c[["X_Card", "X_Action", "Persona",
                                "X_Model"]].rename(columns={"X_Card": "Card",
                                                             "X_Action": "Action"}))
        c2 = df[~df["Fallback"]].copy()
        c2["Persona"] = c2["Persona_Y"]
        fy = c2[c2["X_Action"] == 2][["Y_Card", "Y_Action", "Persona",
                                       "Y_Model"]].rename(
            columns={"Y_Card": "Card", "Y_Action": "Action",
                     "Y_Model": "Model"}).dropna(subset=["Action"])
        all_y_frames.append(fy)

    if all_x_frames:
        pool_x = pd.concat(all_x_frames, ignore_index=True)
        pool_x["Y_bin"] = (pool_x["Action"] == 2).astype(int)
        res_px = _logit(pool_x, "Y_bin ~ C(Card, Treatment('Q')) + "
                                "C(Persona, Treatment('sans_persona'))")
        _print_logit(res_px, "X poolé → P(Miser | Carte + Persona)")
        collected["pool_x"] = {"res": res_px}

    if all_y_frames:
        pool_y = pd.concat(all_y_frames, ignore_index=True)
        pool_y["Y_bin"] = (pool_y["Action"] == 4).astype(int)
        res_py = _logit(pool_y, "Y_bin ~ C(Card, Treatment('Q')) + "
                                "C(Persona, Treatment('sans_persona'))")
        _print_logit(res_py, "Y poolé → P(Suivre | Carte + Persona)")
        collected["pool_y"] = {"res": res_py}

    # --- Par modèle agrégé ---
    print("\n" + "=" * 80)
    print("  RÉGRESSIONS PAR MODÈLE (exp3 agrégé)")
    print("=" * 80)

    all_exp3 = pd.concat(dfs_exp3.values(), ignore_index=True)
    clean_all = all_exp3[~all_exp3["Fallback"]].copy()

    for model_name in clean_all["X_Model"].unique():
        mname = model_name.split("/")[-1]
        print(f"\n  ── {mname} en position X ──")
        sub = (clean_all[clean_all["X_Model"] == model_name]
               [["X_Card", "X_Action", "Persona_X"]]
               .rename(columns={"X_Card": "Card", "Persona_X": "Persona"}))
        sub["Y_bin"] = (sub["X_Action"] == 2).astype(int)
        # Dans exp3 seule, 'sans_persona' n'existe pas → ref = 'rationnel'
        personas = sub["Persona"].unique().tolist()
        ref_p = "rationnel" if "rationnel" in personas else personas[0]
        res = _logit(sub, f"Y_bin ~ C(Card, Treatment('Q')) + "
                          f"C(Persona, Treatment('{ref_p}'))")
        _print_logit(res, f"P(Miser | Carte + Persona) — {mname} [ref={ref_p}]")

    for model_name in clean_all["Y_Model"].unique():
        mname = model_name.split("/")[-1]
        print(f"\n  ── {mname} en position Y ──")
        sub = (clean_all[(clean_all["Y_Model"] == model_name) &
                         (clean_all["X_Action"] == 2)]
               [["Y_Card", "Y_Action", "Persona_Y"]]
               .rename(columns={"Y_Card": "Card", "Persona_Y": "Persona"})
               .dropna(subset=["Y_Action"]))
        sub["Y_bin"] = (sub["Y_Action"] == 4).astype(int)
        # Dans exp3 seule, 'sans_persona' n'existe pas → ref = 'rationnel'
        personas = sub["Persona"].unique().tolist()
        ref_p = "rationnel" if "rationnel" in personas else personas[0]
        res = _logit(sub, f"Y_bin ~ C(Card, Treatment('Q')) + "
                          f"C(Persona, Treatment('{ref_p}'))")
        _print_logit(res, f"P(Suivre | Carte + Persona) — {mname} [ref={ref_p}]")

    return collected

# ==============================================================================
# 6. GRAPHIQUES
# ==============================================================================

def plot_kpi_tables(
    all_kpis: dict[int, dict],
    dfs: dict[int, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Tableaux KPI offense + defense — même style que analyze_v2."""
    for table_type in ("offense", "defense"):
        is_off = (table_type == "offense")
        title  = ("Tableau KPI — Joueur X (métriques offensives) — Expérience 3"
                  if is_off else
                  "Tableau KPI — Joueur Y (métriques défensives) — Expérience 3")

        col_labels = (
            ["Scénario", "Modèle X\n[Persona X]", "N\nvalides",
             "Bluff(J)\nNash≈33%", "ValueBet(K)\nNash=100%",
             "Bet%", "WinRate", "Gain net"]
            if is_off else
            ["Scénario", "Modèle Y\n[Persona Y]", "N\nvalides",
             "Call(J)\nNash=0%", "Call(Q)\nNash≈33%",
             "Call(K)\nNash=100%", "Gain net"]
        )
        col_widths = [3.0, 2.2, 0.9, 1.1, 1.3, 1.0, 1.0, 1.0] if is_off \
                     else [3.0, 2.2, 0.9, 1.1, 1.1, 1.1, 1.0]
        total_w  = sum(col_widths)
        n_rows   = len(all_kpis)
        row_h, header_h = 0.6, 0.8
        fig_h    = header_h + n_rows * row_h + 1.2

        fig, ax = plt.subplots(figsize=(total_w, fig_h))
        ax.axis("off")
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

        x_starts    = np.cumsum([0] + col_widths[:-1]) / total_w
        col_centers = x_starts + np.array(col_widths) / total_w / 2
        y_header    = 1.0 - header_h / fig_h * 0.6
        sep_y       = 1.0 - header_h / fig_h

        for cx, lab in zip(col_centers, col_labels):
            ax.text(cx, y_header, lab, ha="center", va="center",
                    fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.plot([0, 1], [sep_y, sep_y], color="#333333", linewidth=1.0,
                transform=ax.transAxes, clip_on=False)

        for row_i, (sid, k) in enumerate(sorted(all_kpis.items())):
            xm, ym = _model_names(dfs[sid])
            px = dfs[sid]["Persona_X"].iloc[0]
            py = dfs[sid]["Persona_Y"].iloc[0]
            hs = int(dfs[sid]["History_Size"].iloc[0])
            bg = "#F8F8F8" if row_i % 2 == 0 else "#FFFFFF"
            y_row = sep_y - (row_i + 0.5) * row_h / fig_h

            rect = mpatches.FancyBboxPatch(
                (0, sep_y - (row_i + 1) * row_h / fig_h),
                1.0, row_h / fig_h,
                boxstyle="square,pad=0", facecolor=bg, edgecolor="none",
                transform=ax.transAxes, zorder=0,
            )
            ax.add_patch(rect)

            if is_off:
                cells = [
                    (f"S{sid}",              "#222222", False),
                    (f"{xm}\n[{px}]",        PERSONA_COLORS.get(px, "#333"), False),
                    (str(k["n_clean"]),       "#222222", False),
                    (_fmt(k["bluff_rate_x"]), _nash_color(k["bluff_rate_x"], 1/3), True),
                    (_fmt(k["value_bet_x"]),  _nash_color(k["value_bet_x"],  1.0), True),
                    (_fmt(k["bet_rate_x"]),   "#444444", False),
                    (_fmt(k["x_win_rate"]),   "#444444", False),
                    (_fmt(k["x_total_gain"], False),
                     "#1B5E20" if k["x_total_gain"] >= 0 else "#B71C1C", False),
                ]
            else:
                cells = [
                    (f"S{sid}",              "#222222", False),
                    (f"{ym}\n[{py}]",        PERSONA_COLORS.get(py, "#333"), False),
                    (str(k["n_clean"]),       "#222222", False),
                    (_fmt(k["call_rate_j"]),  _nash_color(k["call_rate_j"], 0.0), True),
                    (_fmt(k["call_rate_q"]),  _nash_color(k["call_rate_q"], 1/3), True),
                    (_fmt(k["call_rate_k"]),  _nash_color(k["call_rate_k"], 1.0), True),
                    (_fmt(k["y_total_gain"], False),
                     "#1B5E20" if k["y_total_gain"] >= 0 else "#B71C1C", False),
                ]

            for cx, (txt, color, bold) in zip(col_centers, cells):
                ax.text(cx, y_row, txt, ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight="bold" if bold else "normal",
                        transform=ax.transAxes)

        ax.text(0.5, 0.01,
                "Couleur : vert = proche Nash | orange = écart modéré | rouge = loin",
                ha="center", va="bottom", fontsize=7.5, color=NASH_COLOR,
                style="italic", transform=ax.transAxes)

        _save(fig, figures_dir, f"kpi_{table_type}_table")


def plot_profit_curves(
    dfs: dict[int, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Courbes de profit par scénario, avec légende modèle + persona."""
    n     = len(dfs)
    cols  = min(n, 2)
    rows  = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    fig.suptitle("Gains cumulés nets — Expérience 3 (effet des personas)",
                 fontsize=12, fontweight="bold")

    palette = plt.cm.tab10.colors

    for i, (sid, df) in enumerate(sorted(dfs.items())):
        ax     = axes_flat[i]
        xm, ym = _model_names(df)
        px     = df["Persona_X"].iloc[0]
        py     = df["Persona_Y"].iloc[0]
        ax.set_title(f"S{sid} | {xm}[{px}](X) vs {ym}[{py}](Y)",
                     fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("Partie #", fontsize=9)
        ax.set_ylabel("Gain cumulé net (€)", fontsize=9)
        ax.axhline(0, **NASH_STYLE)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f€"))

        clean = df[~df["Fallback"]].reset_index(drop=True)
        cx    = PERSONA_COLORS.get(px, palette[i % 10])
        cy    = PERSONA_COLORS.get(py, palette[(i+1) % 10])
        games = range(1, len(clean) + 1)
        x_cum = clean["X_Net_Payoff"].cumsum()
        y_cum = clean["Y_Net_Payoff"].cumsum()

        ax.plot(games, x_cum, color=cx, linewidth=1.6,
                label=f"X — {xm} [{px}]")
        ax.plot(games, y_cum, color=cy, linewidth=1.6,
                label=f"Y — {ym} [{py}]")

        if len(clean) > 0:
            for val, color in [(x_cum.iloc[-1], cx), (y_cum.iloc[-1], cy)]:
                ax.annotate(f"{val:+.0f}€", xy=(len(clean), val),
                            fontsize=8, color=color, fontweight="bold",
                            xytext=(5, 0), textcoords="offset points")

        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, figures_dir, "profit_curves")


def plot_persona_comparison(
    dfs_exp3: dict[int, pd.DataFrame],
    dfs_base: dict[int, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """
    Graphique en barres groupées : delta bluff rate et call rate Q
    entre persona (exp3) et baseline (exp1 sans persona).
    Une barre par scénario exp3, couleur = persona.
    Ligne zéro = baseline.
    """
    metrics = ["Δ Bluff(J)", "Δ Call(Q)"]
    scen_data = []

    for sid, df_exp3 in sorted(dfs_exp3.items()):
        df_base = _find_baseline(df_exp3, dfs_base)
        if df_base is None:
            continue

        px = df_exp3["Persona_X"].iloc[0]
        xm = df_exp3["X_Model"].iloc[0].split("/")[-1]
        c3 = df_exp3[~df_exp3["Fallback"]]
        cb = df_base[~df_base["Fallback"]]

        # Bluff rate X
        j3 = c3[c3["X_Card"] == "J"]
        jb = cb[cb["X_Card"] == "J"]
        br3 = float((j3["X_Action"] == 2).mean()) if not j3.empty else float("nan")
        brb = float((jb["X_Action"] == 2).mean()) if not jb.empty else float("nan")

        # Call rate Y avec Q
        q3 = c3[(c3["X_Action"] == 2) & (c3["Y_Card"] == "Q")]
        qb = cb[(cb["X_Action"] == 2) & (cb["Y_Card"] == "Q")]
        cr3 = float((q3["Y_Action"] == 4).mean()) if not q3.empty else float("nan")
        crb = float((qb["Y_Action"] == 4).mean()) if not qb.empty else float("nan")

        scen_data.append({
            "sid":     sid,
            "label":   f"S{sid}\n{xm}\n[{px}]",
            "persona": px,
            "delta_bluff": br3 - brb if not np.isnan(br3 + brb) else float("nan"),
            "delta_callq": cr3 - crb if not np.isnan(cr3 + crb) else float("nan"),
        })

    if not scen_data:
        print("[WARN] Aucune donnée de comparaison — graphique persona ignoré.")
        return

    n   = len(scen_data)
    x   = np.arange(n)
    w   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.5), 5))
    fig.suptitle("Effet des personas — Δ par rapport à la baseline sans persona",
                 fontsize=11, fontweight="bold")

    for ax, metric_key, metric_label, nash_delta in [
        (axes[0], "delta_bluff", "Δ Bluff Rate (J)", 0),
        (axes[1], "delta_callq", "Δ Call Rate (Q)",  0),
    ]:
        vals   = [d[metric_key] for d in scen_data]
        colors = [PERSONA_COLORS.get(d["persona"], "#888888") for d in scen_data]
        labels = [d["label"] for d in scen_data]

        bars = ax.bar(x, vals, width=w * 2, color=colors, alpha=0.85)
        ax.axhline(0, color="#333333", linewidth=1.0, linestyle="-")
        ax.axhline(nash_delta, **NASH_STYLE)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                sign  = "+" if val >= 0 else ""
                color = "#1B5E20" if val <= 0 else "#B71C1C"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005 * np.sign(val + 0.001),
                        f"{sign}{val:.1%}", ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.set_title(metric_label, fontsize=10)

    # Légende personas
    legend_elems = [
        mpatches.Patch(color=PERSONA_COLORS["rationnel"], label="Persona : rationnel"),
        mpatches.Patch(color=PERSONA_COLORS["prudent"],   label="Persona : prudent"),
    ]
    fig.legend(handles=legend_elems, fontsize=8, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    _save(fig, figures_dir, "persona_comparison")


def plot_forest_persona(collected: dict, figures_dir: Path) -> None:
    """
    Forest plot focalisé sur les coefficients Persona
    extraits des régressions poolées.
    OR > 1 = persona augmente P(action) vs sans_persona.
    """
    entries = []

    for key, role, color in [
        ("pool_x", "X → P(Miser)",  FP_COLOR_X),
        ("pool_y", "Y → P(Suivre)", FP_COLOR_Y),
    ]:
        if key not in collected:
            continue
        res = collected[key].get("res", {})
        if "vars" not in res:
            continue
        for v in res["vars"]:
            if "Persona" not in v["label"]:
                continue
            persona_name = (v["label"]
                            .replace("C(Persona, Treatment('sans_persona'))[T.", "")
                            .replace("]", ""))
            pval = v["pval"]
            coef = v["coef"]
            or_v = v["or"]
            sig  = v["sig"]
            if pval > 0 and pval < 1 and not np.isnan(coef):
                z  = sp_stats.norm.ppf(1 - pval / 2) if pval < 1 else 1.96
                z  = max(z, 0.001)
                se = abs(coef) / z
                lo = float(np.exp(coef - 1.96 * se))
                hi = float(np.exp(coef + 1.96 * se))
            else:
                lo, hi = or_v * 0.8, or_v * 1.25

            entries.append({
                "label": f"{role} — Persona {persona_name}\nvs sans persona",
                "or":    or_v,
                "lo95":  lo,
                "hi95":  hi,
                "sig":   sig,
                "color": color,
            })

    if not entries:
        print("[WARN] Aucune entrée pour forest plot persona.")
        return

    n     = len(entries)
    fig_h = max(3.5, 0.6 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y_pos   = list(range(n - 1, -1, -1))

    for entry, yp in zip(entries, y_pos):
        ax.plot([entry["lo95"], entry["hi95"]], [yp, yp],
                color=entry["color"], linewidth=1.4, zorder=2)
        ms = 10 if entry["sig"] != "ns" else 6
        ax.plot(entry["or"], yp, "s", color=entry["color"],
                markersize=ms, zorder=3)
        ax.text(entry["hi95"] * 1.03, yp,
                f" {entry['sig']}  OR={entry['or']:.3f}",
                va="center", fontsize=8, color=entry["color"])

    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (échelle log) — IC 95%\n"
                  "OR > 1 = persona augmente P(action) vs sans persona",
                  fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=8)
    ax.set_title("Forest Plot — Effet causal des personas\n"
                 "(régression poolée exp3 + baseline, ref = sans persona)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)

    legend_elems = [
        mpatches.Patch(color=FP_COLOR_X, label="Joueur X — P(Miser)"),
        mpatches.Patch(color=FP_COLOR_Y, label="Joueur Y — P(Suivre)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
    ax.text(0.01, -0.08,
            "*** p<0.001 | ** p<0.01 | * p<0.05 | ns = non significatif",
            transform=ax.transAxes, fontsize=7.5, color="#555555", style="italic")

    plt.tight_layout()
    _save(fig, figures_dir, "forest_persona")


def plot_forest_scenarios(collected: dict, figures_dir: Path) -> None:
    """Forest plot par scénario + global (même logique qu'analyze_v2)."""
    all_entries = []

    for sid, data in sorted(
        {k: v for k, v in collected.items() if isinstance(k, int)}.items()
    ):
        slbl    = f"S{sid}"
        entries = []

        for res_key, role, color in [
            ("res_x", "X", FP_COLOR_X),
            ("res_y", "Y", FP_COLOR_Y),
        ]:
            res = data.get(res_key, {})
            if "vars" not in res:
                continue
            for v in res["vars"]:
                if "Card" not in v["label"] and "Intercept" not in v["label"]:
                    continue
                if "Intercept" in v["label"]:
                    continue
                card = (v["label"]
                        .replace("C(Card, Treatment('Q'))[T.", "")
                        .replace("]", ""))
                pval, coef, or_v = v["pval"], v["coef"], v["or"]
                sig = v["sig"]
                if pval > 0 and pval < 1 and not np.isnan(coef):
                    z  = max(sp_stats.norm.ppf(1 - pval / 2), 0.001)
                    se = abs(coef) / z
                    lo = float(np.exp(coef - 1.96 * se))
                    hi = float(np.exp(coef + 1.96 * se))
                else:
                    lo, hi = or_v * 0.8, or_v * 1.25
                lbl = f"{role} ({data[role.lower()+'m']}) — Carte {card} vs Q\n[{slbl}]"
                e   = {"label": lbl, "or": or_v, "lo95": lo, "hi95": hi,
                       "sig": sig, "color": color}
                entries.append(e)
                all_entries.append(e)

        if entries:
            _draw_forest(entries,
                         f"Forest Plot — {data['label']}",
                         f"forest_s{sid}", figures_dir)

    if all_entries:
        _draw_forest(all_entries,
                     "Forest Plot global — Tous scénarios exp3",
                     "forest_global", figures_dir)


def _draw_forest(entries: list[dict], title: str,
                 stem: str, figures_dir: Path) -> None:
    if not entries:
        return
    n     = len(entries)
    fig_h = max(3.5, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y_pos   = list(range(n - 1, -1, -1))

    for entry, yp in zip(entries, y_pos):
        ax.plot([entry["lo95"], entry["hi95"]], [yp, yp],
                color=entry["color"], linewidth=1.4, zorder=2)
        ms = 9 if entry["sig"] != "ns" else 6
        ax.plot(entry["or"], yp, "s", color=entry["color"],
                markersize=ms, zorder=3)
        ax.text(entry["hi95"] * 1.02, yp,
                f" {entry['sig']}  OR={entry['or']:.2f}",
                va="center", fontsize=7.5, color=entry["color"])

    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (échelle log) — IC 95%", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=7.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)

    legend_elems = [
        mpatches.Patch(color=FP_COLOR_X, label="Joueur X — P(Miser)"),
        mpatches.Patch(color=FP_COLOR_Y, label="Joueur Y — P(Suivre)"),
    ]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="lower right")
    ax.text(0.01, -0.07,
            "*** p<0.001 | ** p<0.01 | * p<0.05 | ns  |  Référence carte = Q",
            transform=ax.transAxes, fontsize=7, color="#555555", style="italic")

    plt.tight_layout()
    _save(fig, figures_dir, stem)

# ==============================================================================
# 7. POINT D'ENTRÉE
# ==============================================================================

def main() -> None:
    args        = _build_parser().parse_args()
    exp3_dir    = args.exp3
    baseline_dir = args.baseline

    print("\n" + "=" * 80)
    print("  ANALYZE_V3 — Effet des Personas sur la Rationalité Stratégique")
    print(f"  Exp3     : {exp3_dir.resolve()}")
    print(f"  Baseline : {baseline_dir.resolve()}")
    print("=" * 80)

    for d in (exp3_dir, baseline_dir):
        if not d.exists():
            print(f"[ERREUR] Dossier '{d}' introuvable.")
            sys.exit(1)

    figures_dir = exp3_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFigures → {figures_dir.resolve()}")

    # Chargement
    dfs_exp3, dfs_base = load_data(exp3_dir, baseline_dir)
    if not dfs_exp3:
        print("[ERREUR] Aucune donnée exp3.")
        sys.exit(1)

    # KPIs console + graphiques
    all_kpis   = {sid: compute_kpis(df) for sid, df in dfs_exp3.items()}
    summary_df = print_kpi_report(all_kpis, dfs_exp3)
    kpi_csv    = exp3_dir / "report_summary.csv"
    summary_df.to_csv(kpi_csv, index=False, float_format="%.4f")
    print(f"\n[OK] KPIs exportés → {kpi_csv}")

    # Tableau de comparaison exp3 vs baseline
    if dfs_base:
        print_comparison_table(all_kpis, dfs_exp3, dfs_base)
    else:
        print("[WARN] Baseline vide — tableau de comparaison ignoré.")

    # Régressions
    collected = run_regressions(dfs_exp3, dfs_base)

    # Graphiques
    print("\nExport figures...")
    plot_kpi_tables(all_kpis, dfs_exp3, figures_dir)
    plot_profit_curves(dfs_exp3, figures_dir)

    if dfs_base:
        plot_persona_comparison(dfs_exp3, dfs_base, figures_dir)

    plot_forest_persona(collected, figures_dir)
    plot_forest_scenarios(collected, figures_dir)

    print("\n" + "=" * 80)
    print("✓ Analyse terminée.")
    print(f"  Figures : {figures_dir.resolve()}")
    for f in sorted(figures_dir.glob("*")):
        print(f"    {f.name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()