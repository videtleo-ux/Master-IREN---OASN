"""
main_exp3.py — Expérience 3 : Effet des Personas sur la Rationalité Stratégique
=================================================================================
Question de recherche :
    Le comportement observé en exp 1 (hyper-agressivité de Mistral, adaptation
    d'OpenAI) est-il un comportement par défaut modifiable par l'assignation
    d'un persona ? Un persona rationnel fait-il converger les LLMs vers Nash ?

Design expérimental :
    2 modèles × 2 personas × 2 positions = 8 scénarios × 500 parties = 4 000 parties
    Condition de base (sans persona) : déjà disponible via exp 1 (S1, S3)

    Scénario | X_Model  | Y_Model  | Persona_X  | Persona_Y
    ---------|----------|----------|------------|----------
    1        | Mistral  | OpenAI   | rationnel  | rationnel
    2        | Mistral  | OpenAI   | prudent    | prudent
    3        | OpenAI   | Mistral  | rationnel  | rationnel
    4        | OpenAI   | Mistral  | prudent    | prudent
    5        | Mistral  | OpenAI   | rationnel  | prudent   (personas croisés)
    6        | Mistral  | OpenAI   | prudent    | rationnel (personas croisés)
    7        | OpenAI   | Mistral  | rationnel  | prudent   (personas croisés)
    8        | OpenAI   | Mistral  | prudent    | rationnel (personas croisés)

    Note : les scénarios 5-8 (personas croisés) permettent de tester l'interaction
    entre le persona de X et celui de Y — variable indépendante supplémentaire
    sans coût de conception.

Usage :
    python main_exp3.py --scenario [1-8]

Sorties : results_exp3/scenario_{1..8}.csv
    Colonnes identiques à exp 1/2 + deux colonnes supplémentaires :
    Persona_X, Persona_Y

Comparaison avec exp 1 :
    S1 exp3 (Mistral X, rationnel) vs S1 exp1 (Mistral X, sans persona)
    S3 exp3 (OpenAI X, rationnel)  vs S3 exp1 (OpenAI X, sans persona)
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
# 0. CONFIGURATION
# ==============================================================================

MISTRAL_MODEL = "mistral/mistral-small-latest"
OPENAI_MODEL  = "openai/gpt-5.4-mini"

NUM_GAMES     = 500
SLEEP_BETWEEN = 1.0
API_RETRIES   = 3
TOKEN_WARN_THRESHOLD = 4_000

RESULTS_DIR = Path("results_exp3")

# ==============================================================================
# 1. PERSONAS
# ==============================================================================

# Chaque persona est une phrase d'identité insérée en tête du system prompt.
# Volontairement courte et factuelle pour ne pas over-contraindre le modèle.
# Le nom du jeu n'apparaît pas (cohérence avec exp 1 et 2).

PERSONA_DEFINITIONS = {
    "rationnel": (
        "Tu es un agent économique rationnel. "
        "Ton unique objectif est de maximiser ton gain monétaire espéré "
        "sur le long terme en utilisant le raisonnement stratégique optimal."
    ),
    "prudent": (
        "Tu es un joueur naturellement prudent et averse au risque. "
        "Tu préfères éviter les pertes certaines plutôt que de chercher "
        "à maximiser tes gains, et tu hésites à prendre des risques inutiles."
    ),
}

# Table de routage : scenario_id → (x_model, y_model, persona_x, persona_y)
SCENARIOS: dict[int, tuple[str, str, str, str]] = {
    # Personas symétriques
    1: (MISTRAL_MODEL, OPENAI_MODEL,  "rationnel", "rationnel"),
    2: (MISTRAL_MODEL, OPENAI_MODEL,  "prudent",   "prudent"),
    3: (OPENAI_MODEL,  MISTRAL_MODEL, "rationnel", "rationnel"),
    4: (OPENAI_MODEL,  MISTRAL_MODEL, "prudent",   "prudent"),
    # Personas croisés (interaction X persona × Y persona)
    5: (MISTRAL_MODEL, OPENAI_MODEL,  "rationnel", "prudent"),
    6: (MISTRAL_MODEL, OPENAI_MODEL,  "prudent",   "rationnel"),
    7: (OPENAI_MODEL,  MISTRAL_MODEL, "rationnel", "prudent"),
    8: (OPENAI_MODEL,  MISTRAL_MODEL, "prudent",   "rationnel"),
}

# ==============================================================================
# 2. INITIALISATION
# ==============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ACTION_ABATTRE = 1
ACTION_MISER   = 2
ACTION_PASSER  = 3
ACTION_SUIVRE  = 4

CARDS     = ["K", "Q", "J"]
CARD_RANK = {"K": 3, "Q": 2, "J": 1}

CSV_COLUMNS = [
    "Game_ID", "X_Model", "Y_Model", "Persona_X", "Persona_Y",
    "X_Card", "X_Action", "Y_Card", "Y_Action",
    "Winner", "X_Net_Payoff", "Y_Net_Payoff", "Fallback",
]

# ==============================================================================
# 3. LOGIQUE DU JEU
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
# 4. CONSTRUCTION DES PROMPTS AVEC PERSONA
# ==============================================================================

def _action_label(action: Optional[int]) -> str:
    return {
        ACTION_ABATTRE: "Abattre(1)",
        ACTION_MISER:   "Miser(2)",
        ACTION_PASSER:  "Passer(3)",
        ACTION_SUIVRE:  "Suivre(4)",
    }.get(action, "-")  # type: ignore[arg-type]


def build_system_prompt_x(persona: str) -> str:
    """
    Construit le system prompt de X en injectant le persona en tête.
    Le reste du prompt est identique à exp 1/2 pour garantir la comparabilité.
    """
    persona_line = PERSONA_DEFINITIONS[persona]
    return f"""\
{persona_line}

Tu joues EN PREMIER (Joueur X) à un jeu de cartes.

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


def build_system_prompt_y(persona: str) -> str:
    """
    Construit le system prompt de Y en injectant le persona en tête.
    """
    persona_line = PERSONA_DEFINITIONS[persona]
    return f"""\
{persona_line}

Tu joues EN SECOND (Joueur Y) à un jeu de cartes.

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


def build_user_prompt_x(x_card: str) -> str:
    return (
        f"Ta carte : {x_card}.\n\n"
        f"Ton action (réponds uniquement par 1 ou 2) :"
    )


def build_user_prompt_y(y_card: str) -> str:
    return (
        f"Ta carte : {y_card}. L'adversaire a misé.\n\n"
        f"Ton action (réponds uniquement par 3 ou 4) :"
    )


# ==============================================================================
# 5. APPEL LLM ET PARSING
# ==============================================================================

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _parse_action(text: str, valid_actions: set[int]) -> Optional[int]:
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
    estimated = _estimate_tokens(system_prompt + user_prompt)
    if estimated > TOKEN_WARN_THRESHOLD:
        log.warning(
            "Game %d | %s | Prompt estimé ~%d tokens (seuil : %d).",
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
    persona_x: str,
    persona_y: str,
) -> None:
    """
    Exécute NUM_GAMES parties pour le scénario donné.
    Sans historique — design propre pour isoler l'effet causal du persona.
    Reprise automatique si interrompu.
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
        log.info("Reprise scénario %d à la partie %d/%d.",
                 scenario_id, start_game, NUM_GAMES)

    log.info(
        "=== Scénario %d | X=%s [%s] vs Y=%s [%s] | %d parties restantes ===",
        scenario_id,
        x_model.split("/")[-1], persona_x,
        y_model.split("/")[-1], persona_y,
        remaining,
    )

    # Pré-construire les system prompts (constants pour tout le scénario)
    sys_x = build_system_prompt_x(persona_x)
    sys_y = build_system_prompt_y(persona_y)

    for game_num in range(start_game, start_game + remaining):

        x_card, y_card = deal_cards()

        # Action de X
        x_action, x_fallback = call_llm(
            model           = x_model,
            system_prompt   = sys_x,
            user_prompt     = build_user_prompt_x(x_card),
            valid_actions   = {ACTION_ABATTRE, ACTION_MISER},
            fallback_action = ACTION_ABATTRE,
            role_label      = f"X[{persona_x}]",
            game_id         = game_num,
        )

        # Action de Y (uniquement si X a Misé)
        y_action:   Optional[int] = None
        y_fallback: bool          = False

        if x_action == ACTION_MISER:
            y_action, y_fallback = call_llm(
                model           = y_model,
                system_prompt   = sys_y,
                user_prompt     = build_user_prompt_y(y_card),
                valid_actions   = {ACTION_PASSER, ACTION_SUIVRE},
                fallback_action = ACTION_PASSER,
                role_label      = f"Y[{persona_y}]",
                game_id         = game_num,
            )

        winner, x_payoff, y_payoff = compute_payoffs(
            x_card, y_card, x_action, y_action
        )
        is_fallback = x_fallback or y_fallback

        log.info(
            "Game %d/%d | X=%s[%s] Y=%s[%s] | X:%s Y:%s | "
            "Winner=%s | X=%+d€ Y=%+d€%s",
            game_num, NUM_GAMES,
            x_card, x_model.split("/")[-1],
            y_card, y_model.split("/")[-1],
            _action_label(x_action),
            _action_label(y_action) if y_action else "-",
            winner, x_payoff, y_payoff,
            " ⚠ FALLBACK" if is_fallback else "",
        )

        pd.DataFrame([{
            "Game_ID":      game_num,
            "X_Model":      x_model,
            "Y_Model":      y_model,
            "Persona_X":    persona_x,
            "Persona_Y":    persona_y,
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
        prog="main_exp3.py",
        description="Expérience 3 — Effet des personas sur la rationalité stratégique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Scénarios (sans historique — isoler l'effet causal du persona) :

  Personas symétriques :
    1  Mistral(X)[rationnel]  vs OpenAI(Y)[rationnel]
    2  Mistral(X)[prudent]    vs OpenAI(Y)[prudent]
    3  OpenAI(X)[rationnel]   vs Mistral(Y)[rationnel]
    4  OpenAI(X)[prudent]     vs Mistral(Y)[prudent]

  Personas croisés (interaction X × Y) :
    5  Mistral(X)[rationnel]  vs OpenAI(Y)[prudent]
    6  Mistral(X)[prudent]    vs OpenAI(Y)[rationnel]
    7  OpenAI(X)[rationnel]   vs Mistral(Y)[prudent]
    8  OpenAI(X)[prudent]     vs Mistral(Y)[rationnel]

Lancement en parallèle (8 terminaux) :
  python main_exp3.py --scenario 1
  ...
  python main_exp3.py --scenario 8

Comparaison avec exp 1 :
  S1 exp3 (Mistral X, rationnel) vs S1 exp1 (Mistral X, sans persona)
  S3 exp3 (OpenAI X, rationnel)  vs S3 exp1 (OpenAI X, sans persona)
""",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=list(range(1, 9)),
        required=True,
        metavar="[1-8]",
        help="Numéro du scénario à exécuter",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    sid  = args.scenario
    x_model, y_model, persona_x, persona_y = SCENARIOS[sid]

    log.info(
        "Lancement — scénario %d | X=%s [%s] | Y=%s [%s]",
        sid,
        x_model.split("/")[-1], persona_x,
        y_model.split("/")[-1], persona_y,
    )

    run_scenario(
        scenario_id = sid,
        x_model     = x_model,
        y_model     = y_model,
        persona_x   = persona_x,
        persona_y   = persona_y,
    )