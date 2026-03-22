"""
analyze_v2.py — Rapport d'analyse universel + exports graphiques publication
=============================================================================
Usage :
    python analyze_v2.py --dir results        # expérience 1
    python analyze_v2.py --dir results_exp2   # expérience 2

Prérequis :
    pip install pandas matplotlib statsmodels tabulate

Sorties console : IDENTIQUES à analyze.py (tous les tabulate/print conservés).

Sorties fichiers (dans --dir/figures/) :
    profit_curves.png / .pdf       → gains cumulés par scénario
    adaptation.png / .pdf          → évolution stratégique début/fin
    kpi_offense_table.png / .pdf   → tableau KPI Joueur X avec référence Nash
    kpi_defense_table.png / .pdf   → tableau KPI Joueur Y avec référence Nash
    forest_global.png / .pdf       → forest plot global tous scénarios
    forest_sN.png / .pdf           → forest plot par scénario N
    report_summary.csv             → KPIs bruts exportés
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
from tabulate import tabulate

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
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

WINDOW_ADAPT = 100

PALETTE = {
    1: ("#2196F3", "#FF5722"),
    2: ("#1565C0", "#BF360C"),
    3: ("#4CAF50", "#9C27B0"),
    4: ("#2E7D32", "#6A1B9A"),
}

# Couleurs forest plot par rôle
FP_COLOR_X  = "#1565C0"   # bleu foncé — Joueur X
FP_COLOR_Y  = "#BF360C"   # orange foncé — Joueur Y
FP_COLOR_EN = "#2E7D32"   # vert — régression enrichie Y

NASH_COLOR  = "#888888"
NASH_STYLE  = {"color": NASH_COLOR, "linestyle": "--", "linewidth": 1.2, "alpha": 0.7}

# Valeurs Nash théoriques
NASH = {"Bluff(J)": 1/3, "ValueBet(K)": 1.0, "Call(J)": 0.0,
        "Call(Q)": 1/3, "Call(K)": 1.0}

# ==============================================================================
# 1. CLI
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_v2.py",
        description="Analyse universelle des expériences Kuhn Poker LLM — avec exports graphiques publication",
    )
    p.add_argument(
        "--dir", type=Path, default=Path("results"), metavar="DOSSIER",
        help="Dossier contenant les CSV de résultats (défaut : results/)",
    )
    return p

# ==============================================================================
# 2. CHARGEMENT
# ==============================================================================

def load_scenarios(results_dir: Path) -> dict[int, pd.DataFrame]:
    dfs: dict[int, pd.DataFrame] = {}
    csv_files = sorted(results_dir.glob("scenario_*.csv"))
    if not csv_files:
        print(f"[ERREUR] Aucun fichier scenario_*.csv dans '{results_dir}'.")
        sys.exit(1)

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

        if "History_Size" not in df.columns and "Has_History" in df.columns:
            df["History_Size"] = df["Has_History"].apply(lambda v: 20 if v else 0)
        elif "History_Size" not in df.columns:
            df["History_Size"] = 0

        n_fb = df["Fallback"].sum()
        print(f"  [OK] scenario_{sid}.csv : {len(df)} parties ({n_fb} fallbacks)")
        dfs[sid] = df
    return dfs


def _get_label(df: pd.DataFrame, sid: int) -> str:
    x  = df["X_Model"].iloc[0].split("/")[-1] if "X_Model" in df.columns else "X"
    y  = df["Y_Model"].iloc[0].split("/")[-1] if "Y_Model" in df.columns else "Y"
    hs = int(df["History_Size"].iloc[0]) if "History_Size" in df.columns else 0
    hl = f"hist={hs}" if hs > 0 else "sans hist."
    return f"S{sid} — {x}(X) vs {y}(Y) | {hl}"


def _model_names(df: pd.DataFrame) -> tuple[str, str]:
    x = df["X_Model"].iloc[0].split("/")[-1] if "X_Model" in df.columns else "X"
    y = df["Y_Model"].iloc[0].split("/")[-1] if "Y_Model" in df.columns else "Y"
    return x, y


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


def print_kpi_report(all_kpis: dict[int, dict], dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """— SORTIE CONSOLE INCHANGÉE —"""
    rows_x = []
    for sid, k in sorted(all_kpis.items()):
        label = _get_label(dfs[sid], sid)
        rows_x.append([
            label, k["n_clean"], k["n_fallback"],
            f"{k['bluff_rate_x']:.1%}" if not np.isnan(k['bluff_rate_x']) else "N/A",
            f"{k['value_bet_x']:.1%}"  if not np.isnan(k['value_bet_x'])  else "N/A",
            f"{k['bet_rate_x']:.1%}"   if not np.isnan(k['bet_rate_x'])   else "N/A",
            f"{k['x_win_rate']:.1%}"   if not np.isnan(k['x_win_rate'])   else "N/A",
            f"{k['x_total_gain']:+.0f}€",
        ])
    print("\n" + "=" * 80)
    print("  JOUEUR X — Métriques offensives")
    print("=" * 80)
    print(tabulate(rows_x,
        headers=["Scénario","N valides","Fallbacks","Bluff(J)","ValueBet(K)","Bet%","WinRate","Gain net"],
        tablefmt="rounded_outline",
        colalign=("left","right","right","right","right","right","right","right"),
    ))
    print("  Nash théorique : Bluff(J) ≈ 33.3% | ValueBet(K) = 100%")

    rows_y = []
    for sid, k in sorted(all_kpis.items()):
        label = _get_label(dfs[sid], sid)
        rows_y.append([
            label, k["n_clean"],
            f"{k['call_rate_j']:.1%}" if not np.isnan(k['call_rate_j']) else "N/A",
            f"{k['call_rate_q']:.1%}" if not np.isnan(k['call_rate_q']) else "N/A",
            f"{k['call_rate_k']:.1%}" if not np.isnan(k['call_rate_k']) else "N/A",
            f"{k['y_total_gain']:+.0f}€",
        ])
    print("\n" + "=" * 80)
    print("  JOUEUR Y — Métriques défensives (sur mains où X a Misé)")
    print("=" * 80)
    print(tabulate(rows_y,
        headers=["Scénario","N valides","Call(J)","Call(Q)","Call(K)","Gain net"],
        tablefmt="rounded_outline",
        colalign=("left","right","right","right","right","right"),
    ))
    print("  Nash théorique : Call(Q) ≈ 33.3% | Call(K) = 100% | Call(J) = 0%")

    rows_export = []
    for sid, k in sorted(all_kpis.items()):
        rows_export.append({"Scenario_ID": sid, "Label": _get_label(dfs[sid], sid), **k})
    return pd.DataFrame(rows_export)


# ==============================================================================
# 3b. ANALYSE D'EXPLOITATION ADAPTATIVE (scénarios avec historique)
# ==============================================================================
#
# Principe :
#   Dans les scénarios avec historique, les joueurs observent les actions
#   passées et accumulent de l'information empirique sur la stratégie adverse.
#   On mesure ici la capacité d'exploitation adaptative : étant donné ce que
#   chaque joueur a observé de son adversaire, joue-t-il la stratégie qui
#   l'exploite au maximum ?
#
#   Note terminologique : ce n'est PAS du PBE au sens de l'équilibre.
#   Le PBE exige des croyances dérivées de la stratégie d'équilibre (α=1/3).
#   Ici les croyances sont dérivées du comportement empirique observé — c'est
#   de l'apprentissage bayésien dans un jeu répété, plus proche de la notion
#   de stratégie exploitative (Loriente & Diez, 2023).
#
#   (1) Exploitation de Y avec Q : Y doit caller si α_obs > 1/3
#       α_obs = bluff rate empirique de X sur les parties révélées avant N
#
#   (2) Exploitation de X avec J : X doit bluffer si β_obs < 2/3
#       β_obs = call rate empirique de Y sur les parties révélées avant N

def compute_exploitation_analysis(dfs: dict[int, pd.DataFrame]) -> None:
    """
    Calcule et affiche l'analyse d'exploitation adaptative pour les scénarios
    avec historique.

    Pour chaque partie N, reconstruit l'information empirique disponible :
      - α_N = bluff rate observé de X sur les parties révélées avant N
      - β_N = call rate observé de Y sur les parties révélées avant N

    Stratégie exploitative optimale (seuils dérivés des payoffs) :
      - Y avec Q : caller si α_N > 1/3 (seuil d'indifférence de Y)
      - X avec J : bluffer si β_N < 2/3 (seuil d'indifférence de X)

    Mesure : % de parties où chaque joueur joue la stratégie qui exploite
    au maximum l'adversaire observé — indépendamment de l'équilibre de Nash.
    """
    with_hist = {
        sid: df for sid, df in dfs.items()
        if int(df["History_Size"].iloc[0]) > 0
    }
    if not with_hist:
        return

    print("\n" + "=" * 80)
    print("  EXPLOITATION ADAPTATIVE (scénarios avec historique uniquement)")
    print("  Mesure : le joueur joue-t-il la stratégie qui exploite au max l'adversaire observé ?")
    print("  α = bluff rate empirique de X | β = call rate empirique de Y")
    print("  Seuil Y(Q) : caller si α > 1/3 | Seuil X(J) : bluffer si β < 2/3")
    print("=" * 80)

    rows = []
    for sid, df in sorted(with_hist.items()):
        clean  = df[~df["Fallback"]].copy().reset_index(drop=True)
        xm, ym = _model_names(df)
        label  = _get_label(df, sid)
        hs     = int(df["History_Size"].iloc[0])

        # ── BR de Y avec Q ──────────────────────────────────────────────────
        # Pour chaque main où X a misé et Y a Q, calcule α observé avant cette main
        y_q_results = []
        y_faced_q = clean[(clean["X_Action"] == 2) & (clean["Y_Card"] == "Q")]

        for _, row in y_faced_q.iterrows():
            gid    = row["Game_ID"]
            before = clean[clean["Game_ID"] < gid]
            if len(before) < 5:
                continue
            # Parties révélées : X a misé ET Y a suivi (cartes visibles)
            revealed = before[
                (before["X_Action"] == 2) & (before["Y_Action"] == 4)
            ]
            if revealed.empty:
                continue
            alpha = float((revealed["X_Card"] == "J").mean())
            br    = 4 if alpha > 1/3 else 3   # Call si α > 1/3, Fold sinon
            actual = int(row["Y_Action"])
            y_q_results.append({
                "alpha":      alpha,
                "br":         br,
                "actual":     actual,
                "br_correct": (actual == br),
            })

        # ── BR de X avec J ──────────────────────────────────────────────────
        # Pour chaque main où X a J, calcule β observé avant cette main
        x_j_results = []
        x_has_j = clean[clean["X_Card"] == "J"]

        for _, row in x_has_j.iterrows():
            gid    = row["Game_ID"]
            before = clean[clean["Game_ID"] < gid]
            if len(before) < 5:
                continue
            # Parties où X a misé et Y a répondu (call rate observable)
            x_bet = before[before["X_Action"] == 2]
            if x_bet.empty:
                continue
            beta = float((x_bet["Y_Action"] == 4).mean())
            br   = 2 if beta < 2/3 else 1   # Miser si β < 2/3, Abattre sinon
            actual = int(row["X_Action"])
            x_j_results.append({
                "beta":       beta,
                "br":         br,
                "actual":     actual,
                "br_correct": (actual == br),
            })

        # ── Calcul des taux de conformité à la BR ────────────────────────────
        n_y_q   = len(y_q_results)
        n_x_j   = len(x_j_results)
        pbe_y_q = float(np.mean([r["br_correct"] for r in y_q_results])) if n_y_q > 0 else float("nan")
        pbe_x_j = float(np.mean([r["br_correct"] for r in x_j_results])) if n_x_j > 0 else float("nan")

        # α et β moyens observés (pour interpréter la BR théorique dominante)
        alpha_mean = float(np.mean([r["alpha"] for r in y_q_results])) if n_y_q > 0 else float("nan")
        beta_mean  = float(np.mean([r["beta"]  for r in x_j_results])) if n_x_j > 0 else float("nan")

        # BR théorique dominante (celle qui s'applique la plupart du temps)
        br_y_dominant = "Call (α>1/3)" if not np.isnan(alpha_mean) and alpha_mean > 1/3 else "Fold (α<1/3)"
        br_x_dominant = "Miser (β<2/3)" if not np.isnan(beta_mean) and beta_mean < 2/3 else "Abattre (β>2/3)"

        rows.append([
            f"S{sid}",
            f"{xm}(X)\n{ym}(Y)",
            f"hist={hs}",
            f"{alpha_mean:.1%}" if not np.isnan(alpha_mean) else "N/A",
            br_y_dominant,
            f"{pbe_y_q:.1%} (N={n_y_q})" if not np.isnan(pbe_y_q) else "N/A",
            f"{beta_mean:.1%}"  if not np.isnan(beta_mean)  else "N/A",
            br_x_dominant,
            f"{pbe_x_j:.1%} (N={n_x_j})" if not np.isnan(pbe_x_j) else "N/A",
        ])

    print(tabulate(rows,
        headers=[
            "Scén.", "Modèles", "Hist.",
            "α moyen\n(bluff X obs.)", "BR de Y avec Q",  "Y joue sa BR",
            "β moyen\n(call Y obs.)",  "BR de X avec J",  "X joue sa BR",
        ],
        tablefmt="rounded_outline",
        colalign=("center","left","center","right","left","right","right","left","right"),
    ))
    print("  α = bluff rate observé de X sur parties révélées avant chaque main.")
    print("  β = call rate observé de Y sur parties révélées avant chaque main.")
    print("  Exploitation correcte : le joueur joue la stratégie optimale face à l'adversaire observé.")
    print("  Distinction : Nash = stratégie d'équilibre sans info | Exploitation = best-response à l'adversaire réel.")
    print("  Ref : stratégie exploitative au sens de Loriente & Diez (2023).")

# ==============================================================================
# 4. GRAPHIQUES KPI — tableaux matplotlib
# ==============================================================================

def _fmt(v: float, pct: bool = True) -> str:
    if np.isnan(v):
        return "N/A"
    return f"{v:.1%}" if pct else f"{v:+.0f}€"


def _nash_delta_color(val: float, nash: float) -> str:
    if np.isnan(val):
        return "#555555"
    diff = abs(val - nash)
    if diff < 0.05:
        return "#1B5E20"
    if diff < 0.20:
        return "#E65100"
    return "#B71C1C"


def plot_kpi_offense_table(all_kpis: dict[int, dict],
                           dfs: dict[int, pd.DataFrame],
                           figures_dir: Path) -> None:
    """
    Tableau graphique Joueur X.
    Colonnes : Scénario | Modèle X | N valides | Bluff(J) | ValueBet(K) | Bet% | WinRate | Gain net
    Couleur cellule : distance à Nash (vert=proche, rouge=loin).
    """
    col_labels = ["Scénario", "Modèle X", "N\nvalides", "Bluff(J)\nNash≈33%",
                  "ValueBet(K)\nNash=100%", "Bet%\nglobal", "WinRate", "Gain net"]
    col_widths = [3.2, 2.0, 0.9, 1.1, 1.3, 1.0, 1.0, 1.0]
    total_w    = sum(col_widths)
    n_rows     = len(all_kpis)
    row_h      = 0.55
    header_h   = 0.75

    fig_h = header_h + n_rows * row_h + 1.2
    fig, ax = plt.subplots(figsize=(total_w, fig_h))
    ax.axis("off")
    fig.suptitle("Tableau KPI — Joueur X (métriques offensives)",
                 fontsize=12, fontweight="bold", y=0.98)

    # Positionnement des colonnes
    x_starts = np.cumsum([0] + col_widths[:-1]) / total_w
    col_centers = (x_starts + np.array(col_widths) / total_w / 2)

    y_header = 1.0 - header_h / fig_h * 0.6
    for cx, lab in zip(col_centers, col_labels):
        ax.text(cx, y_header, lab, ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                transform=ax.transAxes)

    # Ligne séparatrice header
    sep_y = 1.0 - header_h / fig_h
    ax.plot([0, 1], [sep_y, sep_y], color="#333333", linewidth=1.0,
            transform=ax.transAxes, clip_on=False)

    # Nash references pour coloring
    nash_map = {"bluff_rate_x": 1/3, "value_bet_x": 1.0}

    for row_i, (sid, k) in enumerate(sorted(all_kpis.items())):
        xm, ym = _model_names(dfs[sid])
        hs     = int(dfs[sid]["History_Size"].iloc[0])
        hl     = f"hist={hs}" if hs > 0 else "no hist"
        bg     = "#F8F8F8" if row_i % 2 == 0 else "#FFFFFF"

        y_row = sep_y - (row_i + 0.5) * row_h / fig_h

        # Background row
        rect = mpatches.FancyBboxPatch(
            (0, sep_y - (row_i + 1) * row_h / fig_h),
            1.0, row_h / fig_h,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor="none",
            transform=ax.transAxes, zorder=0,
        )
        ax.add_patch(rect)

        cells = [
            (f"S{sid} | {hl}",            "#222222", False),
            (xm,                           "#1565C0", False),
            (str(k["n_clean"]),            "#222222", False),
            (_fmt(k["bluff_rate_x"]),      _nash_delta_color(k["bluff_rate_x"], 1/3),  True),
            (_fmt(k["value_bet_x"]),       _nash_delta_color(k["value_bet_x"],  1.0),  True),
            (_fmt(k["bet_rate_x"]),        "#444444", False),
            (_fmt(k["x_win_rate"]),        "#444444", False),
            (_fmt(k["x_total_gain"], False),
             "#1B5E20" if k["x_total_gain"] >= 0 else "#B71C1C", False),
        ]
        for cx, (txt, color, bold) in zip(col_centers, cells):
            ax.text(cx, y_row, txt, ha="center", va="center",
                    fontsize=8.5, color=color,
                    fontweight="bold" if bold else "normal",
                    transform=ax.transAxes)

    # Référence Nash en bas
    ax.text(0.5, 0.01,
            "Couleur Bluff/ValueBet : vert = proche Nash | orange = écart modéré | rouge = loin de Nash",
            ha="center", va="bottom", fontsize=7.5, color=NASH_COLOR,
            style="italic", transform=ax.transAxes)

    _save(fig, figures_dir, "kpi_offense_table")


def plot_kpi_defense_table(all_kpis: dict[int, dict],
                           dfs: dict[int, pd.DataFrame],
                           figures_dir: Path) -> None:
    """
    Tableau graphique Joueur Y.
    Colonnes : Scénario | Modèle Y | N valides | Call(J) | Call(Q) | Call(K) | Gain net
    """
    col_labels = ["Scénario", "Modèle Y", "N\nvalides",
                  "Call(J)\nNash=0%", "Call(Q)\nNash≈33%", "Call(K)\nNash=100%", "Gain net"]
    col_widths = [3.2, 2.0, 0.9, 1.1, 1.1, 1.1, 1.0]
    total_w    = sum(col_widths)
    n_rows     = len(all_kpis)
    row_h      = 0.55
    header_h   = 0.75

    fig_h = header_h + n_rows * row_h + 1.2
    fig, ax = plt.subplots(figsize=(total_w, fig_h))
    ax.axis("off")
    fig.suptitle("Tableau KPI — Joueur Y (métriques défensives, mains où X a misé)",
                 fontsize=12, fontweight="bold", y=0.98)

    x_starts    = np.cumsum([0] + col_widths[:-1]) / total_w
    col_centers = (x_starts + np.array(col_widths) / total_w / 2)
    y_header    = 1.0 - header_h / fig_h * 0.6
    sep_y       = 1.0 - header_h / fig_h

    for cx, lab in zip(col_centers, col_labels):
        ax.text(cx, y_header, lab, ha="center", va="center",
                fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax.plot([0, 1], [sep_y, sep_y], color="#333333", linewidth=1.0,
            transform=ax.transAxes, clip_on=False)

    for row_i, (sid, k) in enumerate(sorted(all_kpis.items())):
        xm, ym = _model_names(dfs[sid])
        hs  = int(dfs[sid]["History_Size"].iloc[0])
        hl  = f"hist={hs}" if hs > 0 else "no hist"
        bg  = "#F8F8F8" if row_i % 2 == 0 else "#FFFFFF"
        y_row = sep_y - (row_i + 0.5) * row_h / fig_h

        rect = mpatches.FancyBboxPatch(
            (0, sep_y - (row_i + 1) * row_h / fig_h),
            1.0, row_h / fig_h,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor="none",
            transform=ax.transAxes, zorder=0,
        )
        ax.add_patch(rect)

        cells = [
            (f"S{sid} | {hl}",              "#222222", False),
            (ym,                             "#BF360C", False),
            (str(k["n_clean"]),              "#222222", False),
            (_fmt(k["call_rate_j"]),         _nash_delta_color(k["call_rate_j"], 0.0),  True),
            (_fmt(k["call_rate_q"]),         _nash_delta_color(k["call_rate_q"], 1/3),  True),
            (_fmt(k["call_rate_k"]),         _nash_delta_color(k["call_rate_k"], 1.0),  True),
            (_fmt(k["y_total_gain"], False),
             "#1B5E20" if k["y_total_gain"] >= 0 else "#B71C1C", False),
        ]
        for cx, (txt, color, bold) in zip(col_centers, cells):
            ax.text(cx, y_row, txt, ha="center", va="center",
                    fontsize=8.5, color=color,
                    fontweight="bold" if bold else "normal",
                    transform=ax.transAxes)

    ax.text(0.5, 0.01,
            "Couleur Call : vert = proche Nash | orange = écart modéré | rouge = loin de Nash",
            ha="center", va="bottom", fontsize=7.5, color=NASH_COLOR,
            style="italic", transform=ax.transAxes)

    _save(fig, figures_dir, "kpi_defense_table")

# ==============================================================================
# 5. RÉGRESSION LOGISTIQUE (console inchangée + collecte pour forest plots)
# ==============================================================================

def _logit_card_effect(sub: pd.DataFrame, action_col: str, action_val: int) -> dict:
    data = sub.copy()
    data["Y_bin"] = (data[action_col] == action_val).astype(int)
    if data["Y_bin"].nunique() < 2:
        return {"error": "Variable binaire constante."}
    try:
        model = smf.logit("Y_bin ~ C(Card, Treatment('Q'))", data=data).fit(disp=0)
        res = {"n": len(data), "pseudo_r2": model.prsquared}
        for name, pval, coef in zip(model.pvalues.index, model.pvalues.values, model.params.values):
            cn = name.replace("C(Card, Treatment('Q'))[T.", "").replace("]", "")
            res[f"pval_{cn}"]      = float(pval)
            res[f"coef_{cn}"]      = float(coef)
            res[f"oddsratio_{cn}"] = float(np.exp(coef))
        return res
    except Exception as e:
        return {"error": str(e)}


def _print_regression_result(res: dict, title: str) -> None:
    """— SORTIE CONSOLE INCHANGÉE —"""
    if "error" in res:
        print(f"    {title} : {res['error']}")
        return
    rows = []
    for key, val in res.items():
        if key.startswith("pval_"):
            card   = key.replace("pval_", "")
            or_val = res.get(f"oddsratio_{card}", float("nan"))
            sig    = "***" if val < 0.001 else "**" if val < 0.01 else "*" if val < 0.05 else ""
            rows.append([f"Carte {card} vs Q", f"{val:.4f}", sig, f"{or_val:.2f}"])
    print(f"\n    {title}  (N={res['n']}, pseudo-R²={res['pseudo_r2']:.3f})")
    print(tabulate(rows,
        headers=["Comparaison", "p-value", "Sig.", "Odds Ratio"],
        tablefmt="simple",
        colalign=("left","right","center","right"),
    ))
    print("    Seuils : *** p<0.001 | ** p<0.01 | * p<0.05")


def _build_enriched_y_dataset(clean: pd.DataFrame) -> pd.DataFrame | None:
    rows_y = clean[clean["X_Action"] == 2].copy().reset_index(drop=True)
    if rows_y.empty:
        return None
    results = []
    for _, row in rows_y.iterrows():
        gid    = row["Game_ID"]
        before = clean[clean["Game_ID"] < gid]
        if len(before) < 5:
            continue
        x_rev = before[(before["X_Action"] == 2) & (before["Y_Action"] == 4)]
        x_bluff_obs = float((x_rev["X_Card"] == "J").mean()) if not x_rev.empty else float("nan")
        x_bet_freq  = float((before.tail(20)["X_Action"] == 2).mean())
        results.append({
            "Y_Action":    int(row["Y_Action"]),
            "Y_Card":      row["Y_Card"],
            "x_bluff_obs": x_bluff_obs,
            "x_bet_freq":  x_bet_freq,
            "Game_ID":     gid,
        })
    if not results:
        return None
    return pd.DataFrame(results).dropna(subset=["x_bluff_obs"])


def _run_enriched_regression(enriched: pd.DataFrame) -> dict | None:
    """Console inchangée + retourne le résultat pour forest plot."""
    enriched = enriched.copy()
    enriched["Y_bin"] = (enriched["Y_Action"] == 4).astype(int)
    if enriched["Y_bin"].nunique() < 2 or len(enriched) < 10:
        print("    Y enrichi : données insuffisantes.")
        return None
    try:
        formula = "Y_bin ~ C(Y_Card, Treatment('Q')) + x_bluff_obs + x_bet_freq"
        model   = smf.logit(formula, data=enriched).fit(disp=0)
        rows    = []
        result  = {"n": len(enriched), "pseudo_r2": model.prsquared, "vars": []}
        for name, pval, coef in zip(model.pvalues.index, model.pvalues.values, model.params.values):
            cn = (name.replace("C(Y_Card, Treatment('Q'))[T.", "Card ")
                      .replace("]", "")
                      .replace("x_bluff_obs", "Bluff observé de X")
                      .replace("x_bet_freq",  "Fréquence mise X")
                      .replace("Intercept",   "Constante"))
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            rows.append([cn, f"{pval:.4f}", sig, f"{float(np.exp(coef)):.2f}"])
            result["vars"].append({
                "label": cn, "pval": float(pval), "sig": sig,
                "or": float(np.exp(coef)), "coef": float(coef),
            })
        print(f"\n    Y → P(Suivre | Carte + historique X)  "
              f"(N={len(enriched)}, pseudo-R²={model.prsquared:.3f})")
        print(tabulate(rows,
            headers=["Variable","p-value","Sig.","Odds Ratio"],
            tablefmt="simple",
            colalign=("left","right","center","right"),
        ))
        print("    Seuils : *** p<0.001 | ** p<0.01 | * p<0.05")
        print("    Interp. Odds Ratio > 1 → augmente P(Suivre) | < 1 → diminue")
        return result
    except Exception as e:
        print(f"    Y enrichi : erreur — {e}")
        return None


def run_regressions(dfs: dict[int, pd.DataFrame]) -> dict:
    """
    — SORTIE CONSOLE INCHANGÉE —
    Retourne en plus un dict de résultats structurés pour les forest plots.
    Structure : {sid: {"label", "xm", "ym", "res_x", "res_y", "res_enr"}}
    """
    print("\n" + "=" * 80)
    print("  RÉGRESSIONS LOGISTIQUES")
    print("  Référence carte = Q | p < 0.05 → effet statistiquement significatif")
    print("=" * 80)

    collected: dict[int, dict] = {}

    for sid, df in sorted(dfs.items()):
        clean        = df[~df["Fallback"]].copy()
        label        = _get_label(df, sid)
        history_size = int(df["History_Size"].iloc[0])
        xm, ym       = _model_names(df)

        print(f"\n  ── {label} ──")

        sub_x = clean[["X_Card","X_Action"]].rename(columns={"X_Card":"Card"})
        res_x = _logit_card_effect(sub_x, "X_Action", 2)
        _print_regression_result(res_x, f"X ({xm}) → P(Miser | Carte)")

        sub_y = (clean[clean["X_Action"]==2][["Y_Card","Y_Action"]]
                 .rename(columns={"Y_Card":"Card"})
                 .dropna(subset=["Y_Action"]))
        res_y = _logit_card_effect(sub_y, "Y_Action", 4)
        _print_regression_result(res_y, f"Y ({ym}) → P(Suivre | Carte) [base]")

        res_enr = None
        if history_size > 0:
            enriched = _build_enriched_y_dataset(clean)
            if enriched is not None and len(enriched) > 10:
                res_enr = _run_enriched_regression(enriched)

        collected[sid] = {
            "label": label, "xm": xm, "ym": ym,
            "res_x": res_x, "res_y": res_y, "res_enr": res_enr,
        }

    # --- Par modèle agrégé (console uniquement, inchangé) ---
    print("\n" + "=" * 80)
    print("  RÉGRESSIONS PAR MODÈLE (tous scénarios agrégés)")
    print("=" * 80)
    all_df    = pd.concat(dfs.values(), ignore_index=True)
    clean_all = all_df[~all_df["Fallback"]].copy()

    for model_name in clean_all["X_Model"].unique():
        print(f"\n  ── {model_name.split('/')[-1]} en position X ──")
        sub = clean_all[clean_all["X_Model"]==model_name][["X_Card","X_Action"]].rename(columns={"X_Card":"Card"})
        res = _logit_card_effect(sub, "X_Action", 2)
        _print_regression_result(res, "P(Miser | Carte)")

    for model_name in clean_all["Y_Model"].unique():
        print(f"\n  ── {model_name.split('/')[-1]} en position Y ──")
        sub = (clean_all[(clean_all["Y_Model"]==model_name) & (clean_all["X_Action"]==2)]
               [["Y_Card","Y_Action"]].rename(columns={"Y_Card":"Card"}).dropna(subset=["Y_Action"]))
        res = _logit_card_effect(sub, "Y_Action", 4)
        _print_regression_result(res, "P(Suivre | Carte)")

    return collected

# ==============================================================================
# 6. FOREST PLOTS
# ==============================================================================

def _extract_fp_entries(res: dict, role: str, model_name: str,
                        scenario_label: str, color: str) -> list[dict]:
    """
    Extrait les entrées forest plot depuis un résultat de régression logistique.
    Exclut l'Intercept et les variables constantes.
    Retourne une liste de dicts {label, or, lo95, hi95, pval, sig, color}.
    """
    if not res or "error" in res:
        return []

    entries = []
    # Intervalles de confiance à 95% : OR * exp(±1.96 * SE)
    # On n'a pas SE directement mais on peut le reconstruire via coef et pval
    # Approximation : SE = |coef| / z   où z = norm.ppf(1 - pval/2)
    from scipy import stats as sp_stats

    for key in res:
        if not key.startswith("pval_"):
            continue
        card    = key.replace("pval_", "")
        if card in ("Intercept",):
            continue
        pval    = res[f"pval_{card}"]
        coef    = res.get(f"coef_{card}", float("nan"))
        or_val  = res.get(f"oddsratio_{card}", float("nan"))
        sig     = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

        # IC 95% via z-score
        if not np.isnan(pval) and pval > 0 and pval < 1 and not np.isnan(coef):
            z  = sp_stats.norm.ppf(1 - pval / 2) if pval < 1 else 1.96
            z  = max(z, 0.001)
            se = abs(coef) / z if z > 0 else 0.5
            lo = float(np.exp(coef - 1.96 * se))
            hi = float(np.exp(coef + 1.96 * se))
        else:
            lo, hi = or_val * 0.8, or_val * 1.25

        lbl = f"{role} ({model_name}) — Carte {card} vs Q\n[{scenario_label}]"
        entries.append({
            "label": lbl, "or": or_val, "lo95": lo, "hi95": hi,
            "pval": pval, "sig": sig, "color": color,
        })

    return entries


def _extract_fp_entries_enriched(res_enr: dict, model_name: str,
                                  scenario_label: str) -> list[dict]:
    """Extrait les entrées forest plot depuis la régression Y enrichie."""
    if not res_enr or "vars" not in res_enr:
        return []
    from scipy import stats as sp_stats
    entries = []
    for v in res_enr["vars"]:
        if v["label"] in ("Constante",):
            continue
        pval  = v["pval"]
        coef  = v["coef"]
        or_v  = v["or"]
        sig   = v["sig"]
        if pval > 0 and pval < 1 and not np.isnan(coef):
            z  = sp_stats.norm.ppf(1 - pval / 2) if pval < 1 else 1.96
            z  = max(z, 0.001)
            se = abs(coef) / z if z > 0 else 0.5
            lo = float(np.exp(coef - 1.96 * se))
            hi = float(np.exp(coef + 1.96 * se))
        else:
            lo, hi = or_v * 0.8, or_v * 1.25
        lbl = f"Y-enr ({model_name}) — {v['label']}\n[{scenario_label}]"
        entries.append({
            "label": lbl, "or": or_v, "lo95": lo, "hi95": hi,
            "pval": pval, "sig": sig, "color": FP_COLOR_EN,
        })
    return entries


def _draw_forest(entries: list[dict], title: str,
                 fig_path_stem: str, figures_dir: Path) -> None:
    """
    Dessine et sauvegarde un forest plot.
    Axe x = Odds Ratio (échelle log), ligne verticale OR=1.
    Chaque ligne = une variable, avec son IC 95%, son sig. et sa couleur.
    """
    if not entries:
        print(f"  [SKIP] {fig_path_stem} — aucune donnée valide.")
        return

    n   = len(entries)
    fig_h = max(3.5, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_positions = list(range(n - 1, -1, -1))   # du haut vers le bas

    for i, (entry, yp) in enumerate(zip(entries, y_positions)):
        or_v  = entry["or"]
        lo    = entry["lo95"]
        hi    = entry["hi95"]
        color = entry["color"]
        sig   = entry["sig"]

        # Whisker IC 95%
        ax.plot([lo, hi], [yp, yp], color=color, linewidth=1.4, zorder=2)
        # Carré central (taille proportionnelle à la confiance)
        ms = 9 if sig != "ns" else 6
        ax.plot(or_v, yp, "s", color=color, markersize=ms, zorder=3)

        # Annotation sig + OR
        ax.text(hi * 1.02, yp, f" {sig}  OR={or_v:.2f}", va="center",
                fontsize=7.5, color=color)

    # Ligne OR = 1
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.8, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (échelle log) — IC 95%", fontsize=9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=7.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)

    # Légende couleurs
    legend_elems = [
        mpatches.Patch(color=FP_COLOR_X,  label="Joueur X — P(Miser | Carte)"),
        mpatches.Patch(color=FP_COLOR_Y,  label="Joueur Y — P(Suivre | Carte) [base]"),
        mpatches.Patch(color=FP_COLOR_EN, label="Joueur Y — régression enrichie"),
    ]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="lower right")

    ax.text(0.01, -0.07,
            "Sig. : *** p<0.001 | ** p<0.01 | * p<0.05 | ns = non significatif  "
            "| Référence carte = Q",
            transform=ax.transAxes, fontsize=7, color="#555555", style="italic")

    plt.tight_layout()
    _save(fig, figures_dir, fig_path_stem)


def plot_forest_plots(collected: dict[int, dict], figures_dir: Path) -> None:
    """Génère forest_global + forest_sN pour chaque scénario."""

    all_entries: list[dict] = []

    for sid, data in sorted(collected.items()):
        xm    = data["xm"]
        ym    = data["ym"]
        slbl  = f"S{sid}"

        entries_x   = _extract_fp_entries(data["res_x"],   "X", xm, slbl, FP_COLOR_X)
        entries_y   = _extract_fp_entries(data["res_y"],   "Y", ym, slbl, FP_COLOR_Y)
        entries_enr = _extract_fp_entries_enriched(data.get("res_enr"), ym, slbl)

        scen_entries = entries_x + entries_y + entries_enr
        all_entries += scen_entries

        _draw_forest(
            scen_entries,
            title=f"Forest Plot — {data['label']}",
            fig_path_stem=f"forest_s{sid}",
            figures_dir=figures_dir,
        )

    _draw_forest(
        all_entries,
        title="Forest Plot global — Tous scénarios",
        fig_path_stem="forest_global",
        figures_dir=figures_dir,
    )

# ==============================================================================
# 7. COURBES DE PROFIT (améliorées — identité modèles explicite)
# ==============================================================================

def plot_profit_curves(dfs: dict[int, pd.DataFrame], figures_dir: Path) -> None:
    n     = len(dfs)
    cols  = min(n, 2)
    rows  = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    fig.suptitle("Gains cumulés nets par scénario\n(parties fallback exclues)",
                 fontsize=13, fontweight="bold")

    for i, (sid, df) in enumerate(sorted(dfs.items())):
        ax     = axes_flat[i]
        xm, ym = _model_names(df)
        hs     = int(df["History_Size"].iloc[0])
        hl     = f"hist={hs}" if hs > 0 else "sans historique"
        title  = f"S{sid} | {hl}"

        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Partie #", fontsize=9)
        ax.set_ylabel("Gain cumulé net (€)", fontsize=9)
        ax.axhline(0, **NASH_STYLE)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f€"))

        clean  = df[~df["Fallback"]].reset_index(drop=True)
        cx, cy = PALETTE.get(sid, ("#2196F3", "#FF5722"))
        games  = range(1, len(clean) + 1)
        x_cum  = clean["X_Net_Payoff"].cumsum()
        y_cum  = clean["Y_Net_Payoff"].cumsum()

        ax.plot(games, x_cum, color=cx, linewidth=1.6,
                label=f"X — {xm}")
        ax.plot(games, y_cum, color=cy, linewidth=1.6,
                label=f"Y — {ym}")

        if len(clean) > 0:
            for val, color in [(x_cum.iloc[-1], cx), (y_cum.iloc[-1], cy)]:
                ax.annotate(f"{val:+.0f}€", xy=(len(clean), val),
                            fontsize=8, color=color, fontweight="bold",
                            xytext=(5, 0), textcoords="offset points")

        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, figures_dir, "profit_curves")

# ==============================================================================
# 8. ADAPTATION (console inchangée + export figures amélioré)
# ==============================================================================

def _adaptation_stats(df: pd.DataFrame) -> dict | None:
    clean = df[~df["Fallback"]].reset_index(drop=True)
    if len(clean) < 2 * WINDOW_ADAPT:
        return None

    early = clean.iloc[:WINDOW_ADAPT]
    late  = clean.iloc[-WINDOW_ADAPT:]

    def bluff(sub):
        j = sub[sub["X_Card"] == "J"]
        return float((j["X_Action"] == 2).mean()) if not j.empty else float("nan")
    def vbet(sub):
        k = sub[sub["X_Card"] == "K"]
        return float((k["X_Action"] == 2).mean()) if not k.empty else float("nan")
    def callq(sub):
        f = sub[(sub["X_Action"] == 2) & (sub["Y_Card"] == "Q")]
        return float((f["Y_Action"] == 4).mean()) if not f.empty else float("nan")

    return {
        "bluff_early": bluff(early), "bluff_late": bluff(late),
        "vbet_early":  vbet(early),  "vbet_late":  vbet(late),
        "callq_early": callq(early), "callq_late": callq(late),
    }


def plot_adaptation(dfs: dict[int, pd.DataFrame], figures_dir: Path) -> None:
    with_hist = {sid: df for sid, df in dfs.items()
                 if int(df["History_Size"].iloc[0]) > 0}
    if not with_hist:
        print("[WARN] Aucun scénario avec historique — graphique d'adaptation ignoré.")
        return

    n_valid = [(sid, df, _adaptation_stats(df)) for sid, df in sorted(with_hist.items())]
    n_valid = [(sid, df, st) for sid, df, st in n_valid if st is not None]
    if not n_valid:
        print("[WARN] Données insuffisantes pour l'analyse d'adaptation.")
        return

    metrics   = ["Bluff(J)", "ValueBet(K)", "Call_Q"]
    nash_refs = [1/3, 1.0, 1/3]
    key_pairs = [("bluff_early","bluff_late"),("vbet_early","vbet_late"),
                 ("callq_early","callq_late")]
    bar_w     = 0.25
    x_pos     = [0, 1, 2]

    fig, axes = plt.subplots(1, len(n_valid),
                             figsize=(7.5 * len(n_valid), 6), sharey=False)
    if len(n_valid) == 1:
        axes = [axes]

    fig.suptitle(f"Adaptation stratégique — premières vs dernières {WINDOW_ADAPT} mains\n"
                 "(scénarios avec historique)",
                 fontsize=12, fontweight="bold")

    for ax, (sid, df, st) in zip(axes, n_valid):
        xm, ym = _model_names(df)
        hs     = int(df["History_Size"].iloc[0])
        ax.set_title(f"S{sid} | hist={hs}\n{xm}(X) vs {ym}(Y)",
                     fontsize=9, pad=8)
        cx, cy = PALETTE.get(sid, ("#2196F3", "#FF5722"))

        ev = [st[ek] for ek, _ in key_pairs]
        lv = [st[lk] for _, lk in key_pairs]
        pe = [x - bar_w / 2 for x in x_pos]
        pl = [x + bar_w / 2 for x in x_pos]

        bars_e = ax.bar(pe, ev, bar_w, label=f"Parties 1–{WINDOW_ADAPT}", color=cx, alpha=0.85)
        bars_l = ax.bar(pl, lv, bar_w, label=f"Dernières {WINDOW_ADAPT}", color=cy, alpha=0.85)

        for xp, nash in zip(x_pos, nash_refs):
            ax.hlines(nash, xp - bar_w * 1.2, xp + bar_w * 1.2,
                      colors=NASH_COLOR, linewidths=1.5, linestyles="--",
                      label="Nash" if xp == 0 else "")

        for bar, ev_v, lv_v in zip(bars_l, ev, lv):
            d = lv_v - ev_v
            if not np.isnan(d):
                sign  = "+" if d >= 0 else ""
                color = "#1B5E20" if d >= 0 else "#B71C1C"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"Δ{sign}{d:.1%}", ha="center", va="bottom",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylabel("Taux", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1.25)
        ax.grid(True, axis="y", alpha=0.35, linewidth=0.5)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    _save(fig, figures_dir, "adaptation")

    # --- CONSOLE INCHANGÉE ---
    print("\n" + "=" * 70)
    print("  ADAPTATION STRATÉGIQUE — Évolution début/fin")
    print("=" * 70)
    adapt_rows = []
    for sid, df, st in n_valid:
        for metric, ek, lk, nash in [
            ("Bluff(J)",    "bluff_early", "bluff_late", 1/3),
            ("ValueBet(K)", "vbet_early",  "vbet_late",  1.0),
            ("Call_Q",      "callq_early", "callq_late", 1/3),
        ]:
            e, l = st[ek], st[lk]
            d    = l - e if not (np.isnan(e) or np.isnan(l)) else float("nan")
            closer = (abs(l - nash) < abs(e - nash)) if not np.isnan(d) else None
            interp = ("→ Nash ✓" if closer else "← Nash ✗") if closer is not None else "N/A"
            adapt_rows.append([
                f"S{sid}", metric,
                f"{e:.1%}" if not np.isnan(e) else "N/A",
                f"{l:.1%}" if not np.isnan(l) else "N/A",
                f"{d:+.1%}" if not np.isnan(d) else "N/A",
                interp,
            ])
    print(tabulate(adapt_rows,
        headers=["Scénario","Métrique","Début","Fin","Δ","Tendance"],
        tablefmt="rounded_outline",
        colalign=("center","left","right","right","right","left"),
    ))

# ==============================================================================
# 9. POINT D'ENTRÉE
# ==============================================================================

def main() -> None:
    args        = _build_parser().parse_args()
    results_dir = args.dir

    print("\n" + "=" * 80)
    print(f"  ANALYSE — {results_dir.resolve()}")
    print("=" * 80)

    if not results_dir.exists():
        print(f"[ERREUR] Dossier '{results_dir}' introuvable.")
        sys.exit(1)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFigures → {figures_dir.resolve()}")

    print("\nChargement des données...")
    dfs = load_scenarios(results_dir)

    # --- KPIs : console + graphiques ---
    all_kpis   = {sid: compute_kpis(df) for sid, df in dfs.items()}
    summary_df = print_kpi_report(all_kpis, dfs)

    kpi_csv = results_dir / "report_summary.csv"
    summary_df.to_csv(kpi_csv, index=False, float_format="%.4f")
    print(f"\n[OK] KPIs exportés → {kpi_csv}")

    # --- Analyse d'exploitation adaptative (scénarios avec historique uniquement) ---
    compute_exploitation_analysis(dfs)

    print("\nExport tableaux KPI graphiques...")
    plot_kpi_offense_table(all_kpis, dfs, figures_dir)
    plot_kpi_defense_table(all_kpis, dfs, figures_dir)

    # --- Régressions : console + collecte pour forest plots ---
    collected = run_regressions(dfs)

    # --- Forest plots ---
    print("\nExport forest plots...")
    plot_forest_plots(collected, figures_dir)

    # --- Courbes de profit ---
    print("\nExport courbes de profit...")
    plot_profit_curves(dfs, figures_dir)

    # --- Adaptation ---
    print("\nExport graphique d'adaptation...")
    plot_adaptation(dfs, figures_dir)

    print("\n" + "=" * 80)
    print("✓ Analyse terminée.")
    print(f"  Console    : KPIs + régressions (inchangé)")
    print(f"  Figures    : {figures_dir.resolve()}")
    files = sorted(figures_dir.glob("*"))
    for f in files:
        print(f"    {f.name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()