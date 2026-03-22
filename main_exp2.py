"""
main_exp2.py — Expérience 2 : Self-Play OpenAI avec historique modulable
=========================================================================
Usage :
    python main_exp2.py --scenario 1   # Sans historique       (history_size=0)
    python main_exp2.py --scenario 2   # Historique court      (history_size=20)
    python main_exp2.py --scenario 3   # Historique étendu     (history_size=100)

Sorties : results_exp2/scenario_{1,2,3}.csv

Colonnes CSV identiques à l'exp 1 — compatible avec analyze.py.
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

import litellm
import pandas as pd
from dotenv import load_dotenv

# ==============================================================================
# 0. CONFIGURATION — modifier ici uniquement
# ==============================================================================

OPENAI_MODEL  = "openai/gpt-5.4-mini"   # ← changer pour gpt-5.4-mini quand dispo

NUM_GAMES     = 500     # parties par scénario
SLEEP_BETWEEN = 1.0     # pause entre parties (politesse API)
API_RETRIES   = 3       # tentatives max par appel LLM

# Seuil d'estimation de tokens au-delà duquel un WARNING est émis.
# Estimation grossière : 1 token ≈ 4 caractères.
TOKEN_WARN_THRESHOLD = 4_000

RESULTS_DIR = Path("results_exp2")

# Table de routage : scenario_id → (x_model, y_model, history_size)
# history_size=0 désactive l'historique ; >0 = nb de parties max transmises au LLM.
SCENARIOS: dict[int, tuple[str, str, int]] = {
    1: (OPENAI_MODEL, OPENAI_MODEL,  0),    # sans historique
    2: (OPENAI_MODEL, OPENAI_MODEL, 20),    # historique court
    3: (OPENAI_MODEL, OPENAI_MODEL, 100),   # historique étendu
}

# ==============================================================================
# 1. INITIALISATION
# ==============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constantes d'actions
ACTION_ABATTRE = 1
ACTION_MISER   = 2
ACTION_PASSER  = 3
ACTION_SUIVRE  = 4

CARDS     = ["K", "Q", "J"]
CARD_RANK = {"K": 3, "Q": 2, "J": 1}

CSV_COLUMNS = [
    "Game_ID", "X_Model", "Y_Model", "History_Size",
    "X_Card", "X_Action", "Y_Card", "Y_Action",
    "Winner", "X_Net_Payoff", "Y_Net_Payoff", "Fallback",
]

# ==============================================================================
# 2. LOGIQUE DU JEU
# ==============================================================================

def deal_cards() -> tuple[str, str]:
    sample = random.sample(CARDS, 2)
    return sample[0], sample[1]


def determine_winner(x_card: str, y_card: str) -> str:
    return "X" if CARD_RANK[x_card] > CARD_RANK[y_card] else "Y"


def compute_payoffs(
    x_card: str,
    y_card: str,
    x_action: int,
    y_action: Optional[int],
) -> tuple[str, int, int]:
    if x_action == ACTION_ABATTRE:
        winner = determine_winner(x_card, y_card)
        return (winner, 1, -1) if winner == "X" else (winner, -1, 1)
    if x_action == ACTION_MISER:
        if y_action == ACTION_PASSER:
            return "X", 1, -1
        if y_action == ACTION_SUIVRE:
            winner = determine_winner(x_card, y_card)
            return (winner, 2, -2) if winner == "X" else (winner, -2, 2)
    raise ValueError(f"Combinaison invalide : X={x_action}, Y={y_action}")


# ==============================================================================
# 3. GESTION DE L'HISTORIQUE — BROUILLARD DE GUERRE
# ==============================================================================

def _action_label(action: Optional[int]) -> str:
    return {
        ACTION_ABATTRE: "Abattre(1)",
        ACTION_MISER:   "Miser(2)",
        ACTION_PASSER:  "Passer(3)",
        ACTION_SUIVRE:  "Suivre(4)",
    }.get(action, "-")  # type: ignore[arg-type]


def _estimate_tokens(text: str) -> int:
    """Estimation grossière : 1 token ≈ 4 caractères."""
    return len(text) // 4


def build_history_table_for_x(game_log: list[dict], last_n: int) -> str:
    """
    Tableau Markdown VU PAR X.
    Masquage : si Y=Passer, carte de Y → '?'.
    """
    rows = game_log[-last_n:] if last_n > 0 else []
    if not rows:
        return ""
    lines = [
        "| Game | Your Card | X Action | Y Action | Opponent Card | Result |",
        "|------|-----------|----------|----------|---------------|--------|",
    ]
    for r in rows:
        ya             = r.get("Y_Action")
        y_card_display = r["Y_Card"] if ya != ACTION_PASSER else "?"
        result         = "Win" if r["Winner"] == "X" else "Loss"
        lines.append(
            f"| {r['Game_ID']} | {r['X_Card']} | {_action_label(r['X_Action'])} "
            f"| {_action_label(ya)} | {y_card_display} | {result} |"
        )
    return "\n".join(lines)


def build_history_table_for_y(game_log: list[dict], last_n: int) -> str:
    """
    Tableau Markdown VU PAR Y.
    Masquage : si Y=Passer, carte de X → '?'.
    """
    rows = game_log[-last_n:] if last_n > 0 else []
    if not rows:
        return ""
    lines = [
        "| Game | Your Card | X Action | Y Action | Opponent Card | Result |",
        "|------|-----------|----------|----------|---------------|--------|",
    ]
    for r in rows:
        ya             = r.get("Y_Action")
        x_card_display = r["X_Card"] if ya != ACTION_PASSER else "?"
        result         = "Win" if r["Winner"] == "Y" else "Loss"
        lines.append(
            f"| {r['Game_ID']} | {r['Y_Card']} | {_action_label(r['X_Action'])} "
            f"| {_action_label(ya)} | {x_card_display} | {result} |"
        )
    return "\n".join(lines)


# ==============================================================================
# 4. PROMPTS
# ==============================================================================

SYSTEM_PROMPT_X = """\
Tu es un joueur de cartes. Tu joues EN PREMIER (Joueur X).

Contexte : 3 cartes existent (K > Q > J). Chaque joueur en reçoit 1. Pot : 2€ (1€ d'ante chacun).

Tes deux options :
  1 = Abattre  → Révéler immédiatement les deux cartes. La meilleure main gagne ±1€.
  2 = Miser    → Tu ajoutes 1€ au pot. Ton adversaire devra ensuite :
                   - Passer (3) : il se couche, tu gagnes +1€ SANS révéler ta carte.
                   - Suivre (4) : il ajoute 1€, révélation, le meilleur gagne ±2€.

FORMAT DE RÉPONSE OBLIGATOIRE : un seul caractère, soit "1" soit "2".
Exemples de réponses valides : 1 | 2
Exemples de réponses INVALIDES : "Je choisis 1", "Abattre", "bet", "check", "1."

Réponds maintenant avec un unique chiffre (1 ou 2) et rien d'autre.\
"""

SYSTEM_PROMPT_Y = """\
Tu es un joueur de cartes. Tu joues EN SECOND (Joueur Y).

Contexte : 3 cartes existent (K > Q > J). Chaque joueur en reçoit 1. Pot : 2€ (1€ d'ante chacun).
Joueur X jouait EN PREMIER. Il avait deux options : Abattre (1) ou Miser (2).
Il a choisi de MISER (il a ajouté 1€ au pot).

TES DEUX ACTIONS POSSIBLES :
  ACTION "3" = PASSER  → Tu abandonnes la main. Résultat net pour toi : -1€.
  ACTION "4" = SUIVRE  → Tu égalises (ajoutes 1€). Les cartes sont révélées. Le meilleur gagne ±2€.

FORMAT DE RÉPONSE OBLIGATOIRE : un seul caractère, soit "3" soit "4".
Exemples de réponses valides : 3 | 4
Exemples de réponses INVALIDES : "Je choisis 3", "Passer", "fold", "call", "3."

Réponds maintenant avec un unique chiffre (3 ou 4) et rien d'autre.\
"""


def build_user_prompt_x(x_card: str, history_md: str, history_size: int) -> str:
    hist_section = (
        f"\n\nHistorique de tes {history_size} dernières parties :\n{history_md}"
        if history_md else ""
    )
    return (
        f"Ta carte : {x_card}.{hist_section}\n\n"
        f"Ton action (réponds uniquement par 1 ou 2) :"
    )


def build_user_prompt_y(y_card: str, history_md: str, history_size: int) -> str:
    hist_section = (
        f"\n\nHistorique de tes {history_size} dernières parties :\n{history_md}"
        if history_md else ""
    )
    return (
        f"Ta carte : {y_card}. L'adversaire a misé.{hist_section}\n\n"
        f"Ton action (réponds uniquement par 3 ou 4) :"
    )


# ==============================================================================
# 5. APPEL LLM ET PARSING ROBUSTE
# ==============================================================================

def _parse_action(text: str, valid_actions: set[int]) -> Optional[int]:
    """Parsing en 2 passes : word-boundary puis premier chiffre brut."""
    cleaned = text.strip()
    for m in re.findall(r"\b([1-4])\b", cleaned):
        action = int(m)
        if action in valid_actions:
            return action
    for ch in cleaned:
        if ch.isdigit():
            action = int(ch)
            if action in valid_actions:
                return action
    return None


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    valid_actions: set[int],
    fallback_action: int,
    role_label: str,
    game_id: int,
) -> tuple[int, bool]:
    """
    Appelle le LLM avec robustesse en 3 niveaux.
    Émet un WARNING si le prompt dépasse TOKEN_WARN_THRESHOLD tokens estimés.
    """
    estimated = _estimate_tokens(system_prompt + user_prompt)
    if estimated > TOKEN_WARN_THRESHOLD:
        log.warning(
            "Game %d | %s | Prompt estimé à ~%d tokens (seuil : %d) — "
            "coût élevé possible.",
            game_id, role_label, estimated, TOKEN_WARN_THRESHOLD,
        )

    for attempt in range(1, API_RETRIES + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=30,
                temperature=1.0,
                num_retries=API_RETRIES,
            )
            raw_text = response.choices[0].message.content or ""
            action   = _parse_action(raw_text, valid_actions)

            if action is not None:
                return action, False

            log.warning(
                "Game %d | %s (%s) | Réponse invalide (tentative %d/%d) : %r",
                game_id, role_label, model, attempt, API_RETRIES, raw_text,
            )

        except Exception as exc:
            log.warning(
                "Game %d | %s (%s) | Erreur API (tentative %d/%d) : %s",
                game_id, role_label, model, attempt, API_RETRIES, exc,
            )

    log.error(
        "Game %d | %s (%s) | Échec après %d tentatives → FALLBACK action=%d",
        game_id, role_label, model, API_RETRIES, fallback_action,
    )
    return fallback_action, True


# ==============================================================================
# 6. BOUCLE DE JEU
# ==============================================================================

def run_scenario(
    scenario_id: int,
    x_model: str,
    y_model: str,
    history_size: int,
) -> None:
    """
    Exécute NUM_GAMES parties.
    Reprise automatique si interrompu.
    Écriture CSV ligne par ligne (mode append).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"scenario_{scenario_id}.csv"

    if not csv_path.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
        log.info("Fichier créé : %s", csv_path)

    existing   = pd.read_csv(csv_path)
    start_game = int(existing["Game_ID"].max()) + 1 if not existing.empty else 1
    remaining  = NUM_GAMES - (start_game - 1)

    if remaining <= 0:
        log.info("Scénario %d déjà complet. Rien à faire.", scenario_id)
        return

    if start_game > 1:
        log.info("Reprise scénario %d à la partie %d/%d.", scenario_id, start_game, NUM_GAMES)

    hist_label = f"history_size={history_size}" if history_size > 0 else "sans historique"
    log.info(
        "=== Scénario %d | %s vs %s | %s | %d parties restantes ===",
        scenario_id, x_model.split("/")[-1], y_model.split("/")[-1],
        hist_label, remaining,
    )

    game_log: list[dict] = existing.to_dict("records") if not existing.empty else []

    for game_num in range(start_game, start_game + remaining):

        x_card, y_card = deal_cards()

        history_x_md = build_history_table_for_x(game_log, history_size)
        history_y_md = build_history_table_for_y(game_log, history_size)

        x_action, x_fallback = call_llm(
            model           = x_model,
            system_prompt   = SYSTEM_PROMPT_X,
            user_prompt     = build_user_prompt_x(x_card, history_x_md, history_size),
            valid_actions   = {ACTION_ABATTRE, ACTION_MISER},
            fallback_action = ACTION_ABATTRE,
            role_label      = "X",
            game_id         = game_num,
        )

        y_action:   Optional[int] = None
        y_fallback: bool          = False

        if x_action == ACTION_MISER:
            y_action, y_fallback = call_llm(
                model           = y_model,
                system_prompt   = SYSTEM_PROMPT_Y,
                user_prompt     = build_user_prompt_y(y_card, history_y_md, history_size),
                valid_actions   = {ACTION_PASSER, ACTION_SUIVRE},
                fallback_action = ACTION_PASSER,
                role_label      = "Y",
                game_id         = game_num,
            )

        winner, x_payoff, y_payoff = compute_payoffs(x_card, y_card, x_action, y_action)
        is_fallback = x_fallback or y_fallback

        log.info(
            "Game %d/%d | X=%s[%s] Y=%s[%s] | X:%s Y:%s | Winner=%s | X=%+d€ Y=%+d€%s",
            game_num, NUM_GAMES,
            x_card, x_model.split("/")[-1],
            y_card, y_model.split("/")[-1],
            _action_label(x_action),
            _action_label(y_action) if y_action else "-",
            winner, x_payoff, y_payoff,
            " ⚠ FALLBACK" if is_fallback else "",
        )

        game_log.append({
            "Game_ID":      game_num,
            "X_Card":       x_card,
            "X_Action":     x_action,
            "Y_Card":       y_card,
            "Y_Action":     y_action,
            "Winner":       winner,
            "X_Net_Payoff": x_payoff,
            "Y_Net_Payoff": y_payoff,
        })

        pd.DataFrame([{
            "Game_ID":      game_num,
            "X_Model":      x_model,
            "Y_Model":      y_model,
            "History_Size": history_size,
            "X_Card":       x_card,
            "X_Action":     x_action,
            "Y_Card":       y_card,
            "Y_Action":     y_action if y_action is not None else "",
            "Winner":       winner,
            "X_Net_Payoff": x_payoff,
            "Y_Net_Payoff": y_payoff,
            "Fallback":     is_fallback,
        }]).to_csv(csv_path, mode="a", header=False, index=False)

        time.sleep(SLEEP_BETWEEN)

    log.info("=== Scénario %d terminé → %s ===", scenario_id, csv_path)


# ==============================================================================
# 7. POINT D'ENTRÉE CLI
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_exp2.py",
        description="Expérience 2 — Self-Play OpenAI avec historique modulable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Scénarios :
  1  Sans historique       (history_size=0)
  2  Historique court      (history_size=20)
  3  Historique étendu     (history_size=100)

Lancement en parallèle (3 terminaux) :
  python main_exp2.py --scenario 1
  python main_exp2.py --scenario 2
  python main_exp2.py --scenario 3
""",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        required=True,
        metavar="[1-3]",
        help="Numéro du scénario à exécuter",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    sid  = args.scenario
    x_model, y_model, history_size = SCENARIOS[sid]

    log.info(
        "Lancement — scénario %d | %s vs %s | history_size=%d",
        sid, x_model, y_model, history_size,
    )
    run_scenario(
        scenario_id  = sid,
        x_model      = x_model,
        y_model      = y_model,
        history_size = history_size,
    )
