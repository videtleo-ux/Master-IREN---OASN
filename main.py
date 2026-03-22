"""
main.py — Évaluation de la rationalité des LLMs au Kuhn Poker
==============================================================
Usage :
    python main.py --scenario 1   # Mistral(X) vs OpenAI(Y) | sans historique
    python main.py --scenario 2   # Mistral(X) vs OpenAI(Y) | avec historique
    python main.py --scenario 3   # OpenAI(X)  vs Mistral(Y) | sans historique
    python main.py --scenario 4   # OpenAI(X)  vs Mistral(Y) | avec historique

Architecture :
    .env              → MISTRAL_API_KEY, OPENAI_API_KEY
    requirements.txt  → litellm, pandas, python-dotenv, matplotlib
    main.py           → ce fichier
    analyze.py        → rapport d'analyse post-expérience
    results/          → CSV par scénario (créés automatiquement)
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
# 0. CONFIGURATION GLOBALE — seul endroit à modifier pour changer les modèles
# ==============================================================================

MISTRAL_MODEL = "mistral/mistral-small-latest"
OPENAI_MODEL  = "openai/gpt-5.4-mini"           # remplacer par gpt-5.4-mini quand dispo

NUM_GAMES     = 500      # parties par scénario
HISTORY_SIZE  = 20       # nb de parties max dans l'historique fourni au LLM
SLEEP_BETWEEN = 1.0      # secondes de pause entre chaque partie (politesse API)
API_RETRIES   = 3        # tentatives max en cas de réponse invalide ou erreur réseau

RESULTS_DIR = Path("results")

# Table de routage des scénarios : id → (x_model, y_model, has_history)
SCENARIOS: dict[int, tuple[str, str, bool]] = {
    1: (MISTRAL_MODEL, OPENAI_MODEL,  False),
    2: (MISTRAL_MODEL, OPENAI_MODEL,  True),
    3: (OPENAI_MODEL,  MISTRAL_MODEL, False),
    4: (OPENAI_MODEL,  MISTRAL_MODEL, True),
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
ACTION_ABATTRE = 1   # X : révéler les cartes immédiatement (check/showdown)
ACTION_MISER   = 2   # X : miser 1€ supplémentaire (bet)
ACTION_PASSER  = 3   # Y : se coucher (fold)
ACTION_SUIVRE  = 4   # Y : suivre la mise (call)

# Hiérarchie des cartes
CARDS     = ["K", "Q", "J"]
CARD_RANK = {"K": 3, "Q": 2, "J": 1}

# Colonnes du fichier CSV de résultats
CSV_COLUMNS = [
    "Game_ID", "X_Model", "Y_Model", "Has_History",
    "X_Card", "X_Action", "Y_Card", "Y_Action",
    "Winner", "X_Net_Payoff", "Y_Net_Payoff", "Fallback",
]

# ==============================================================================
# 2. LOGIQUE DU JEU
# ==============================================================================

def deal_cards() -> tuple[str, str]:
    """Distribue aléatoirement 2 cartes parmi 3 (la 3e reste non distribuée)."""
    sample = random.sample(CARDS, 2)
    return sample[0], sample[1]  # (x_card, y_card)


def determine_winner(x_card: str, y_card: str) -> str:
    """Retourne 'X' ou 'Y' selon la force des cartes (K > Q > J)."""
    return "X" if CARD_RANK[x_card] > CARD_RANK[y_card] else "Y"


def compute_payoffs(
    x_card: str,
    y_card: str,
    x_action: int,
    y_action: Optional[int],
) -> tuple[str, int, int]:
    """
    Calcule (winner, x_net_payoff, y_net_payoff) selon les règles du Kuhn Poker.

    Séquences possibles :
      X=1              → showdown immédiat, gagnant ±1€
      X=2, Y=3         → X gagne sans révéler, ±1€
      X=2, Y=4         → showdown, gagnant ±2€
    """
    if x_action == ACTION_ABATTRE:
        winner = determine_winner(x_card, y_card)
        return (winner, 1, -1) if winner == "X" else (winner, -1, 1)

    if x_action == ACTION_MISER:
        if y_action == ACTION_PASSER:
            return "X", 1, -1
        if y_action == ACTION_SUIVRE:
            winner = determine_winner(x_card, y_card)
            return (winner, 2, -2) if winner == "X" else (winner, -2, 2)

    raise ValueError(f"Combinaison d'actions invalide : X={x_action}, Y={y_action}")


# ==============================================================================
# 3. GESTION DE L'HISTORIQUE — BROUILLARD DE GUERRE
# ==============================================================================

def _action_label(action: Optional[int]) -> str:
    """Convertit un entier d'action en label lisible pour les tableaux Markdown."""
    labels = {
        ACTION_ABATTRE: "Abattre(1)",
        ACTION_MISER:   "Miser(2)",
        ACTION_PASSER:  "Passer(3)",
        ACTION_SUIVRE:  "Suivre(4)",
    }
    return labels.get(action, "-")  # type: ignore[arg-type]


def build_history_table_for_x(game_log: list[dict], last_n: int = HISTORY_SIZE) -> str:
    """
    Génère le tableau Markdown des `last_n` dernières parties VU PAR X.

    Règle de masquage : si Y a Passé (action 3), X ne connaît pas la carte de Y → '?'.
    Si X=1 ou Y=4, les deux cartes ont été révélées → affichage en clair.
    """
    rows = game_log[-last_n:]
    if not rows:
        return ""

    lines = [
        "| Game | Your Card | X Action | Y Action | Opponent Card | Result |",
        "|------|-----------|----------|----------|---------------|--------|",
    ]
    for r in rows:
        y_action       = r.get("Y_Action")
        y_card_display = r["Y_Card"] if y_action != ACTION_PASSER else "?"
        y_action_label = _action_label(y_action)
        result         = "Win" if r["Winner"] == "X" else "Loss"
        lines.append(
            f"| {r['Game_ID']} "
            f"| {r['X_Card']} "
            f"| {_action_label(r['X_Action'])} "
            f"| {y_action_label} "
            f"| {y_card_display} "
            f"| {result} |"
        )
    return "\n".join(lines)


def build_history_table_for_y(game_log: list[dict], last_n: int = HISTORY_SIZE) -> str:
    """
    Génère le tableau Markdown des `last_n` dernières parties VU PAR Y.

    Règle de masquage : si Y a Passé (action 3), Y ne connaît pas la carte de X → '?'.
    Si X=1 ou Y=4, les deux cartes ont été révélées → affichage en clair.
    """
    rows = game_log[-last_n:]
    if not rows:
        return ""

    lines = [
        "| Game | Your Card | X Action | Y Action | Opponent Card | Result |",
        "|------|-----------|----------|----------|---------------|--------|",
    ]
    for r in rows:
        y_action       = r.get("Y_Action")
        x_card_display = r["X_Card"] if y_action != ACTION_PASSER else "?"
        y_action_label = _action_label(y_action)
        result         = "Win" if r["Winner"] == "Y" else "Loss"
        lines.append(
            f"| {r['Game_ID']} "
            f"| {r['Y_Card']} "
            f"| {_action_label(r['X_Action'])} "
            f"| {y_action_label} "
            f"| {x_card_display} "
            f"| {result} |"
        )
    return "\n".join(lines)


# ==============================================================================
# 4. PROMPTS SYSTÈME ET UTILISATEUR
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


def build_user_prompt_x(x_card: str, history_md: str) -> str:
    hist_section = (
        f"\n\nHistorique de tes {HISTORY_SIZE} dernières parties :\n{history_md}"
        if history_md else ""
    )
    return (
        f"Ta carte : **{x_card}**."
        f"{hist_section}\n\n"
        f"Quelle est ton action ? (1 = Abattre / 2 = Miser)"
    )


def build_user_prompt_y(y_card: str, history_md: str) -> str:
    hist_section = (
        f"\n\nHistorique de tes {HISTORY_SIZE} dernières parties :\n{history_md}"
        if history_md else ""
    )
    return (
        f"Ta carte : {y_card}. L'adversaire a misé."
        f"{hist_section}\n\n"
        f"Ton action (réponds uniquement par 3 ou 4) :"
    )


# ==============================================================================
# 5. APPEL LLM ET PARSING ROBUSTE
# ==============================================================================

def _parse_action(text: str, valid_actions: set[int]) -> Optional[int]:
    """
    Extrait le premier entier valide de la réponse brute du LLM.

    Stratégie en 2 passes pour absorber les réponses tronquées ou verboses :
      1. Recherche d'un chiffre isolé (word-boundary) — cas normal.
      2. Recherche du premier chiffre valide dans la chaîne — cas réponse tronquée
         (ex: "3." ou "3\n" ou "3 -" générés quand max_tokens est atteint mi-phrase).
    """
    cleaned = text.strip()

    # Passe 1 : chiffre isolé (ex : "3", "Réponse : 4")
    for m in re.findall(r"\b([1-4])\b", cleaned):
        action = int(m)
        if action in valid_actions:
            return action

    # Passe 2 : premier chiffre valide dans la chaîne brute (réponse tronquée)
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
    Appelle le LLM et retourne (action, is_fallback).

    Stratégie de robustesse en 3 niveaux :
      1. litellm gère les erreurs réseau via num_retries (backoff exponentiel auto).
      2. Si la réponse n'est pas parseable → on ré-envoie le prompt (jusqu'à API_RETRIES).
      3. Si toujours invalide → action par défaut contextuelle + flag Fallback=True.

    Actions par défaut (passives, ne divulguent aucune information) :
      X → 1 (Abattre)   |   Y → 3 (Passer)
    """
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
# 6. BOUCLE DE JEU — exécution d'un scénario complet
# ==============================================================================

def run_scenario(
    scenario_id: int,
    x_model: str,
    y_model: str,
    has_history: bool,
) -> None:
    """
    Exécute NUM_GAMES parties pour le scénario donné.

    Fonctionnalités :
      - Création automatique du dossier results/ et du fichier CSV avec en-tête.
      - Reprise automatique si le script a été interrompu (détection du dernier Game_ID).
      - Écriture CSV immédiate après chaque partie (mode append, tolérant aux crashes).
    """
    # --- Préparation du répertoire et du fichier ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"scenario_{scenario_id}.csv"

    if not csv_path.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
        log.info("Fichier de résultats créé : %s", csv_path)

    # Lecture de l'existant pour déterminer le point de reprise
    existing   = pd.read_csv(csv_path)
    start_game = int(existing["Game_ID"].max()) + 1 if not existing.empty else 1
    remaining  = NUM_GAMES - (start_game - 1)

    if remaining <= 0:
        log.info(
            "Scénario %d déjà complet (%d/%d parties). Rien à faire.",
            scenario_id, NUM_GAMES, NUM_GAMES,
        )
        return

    if start_game > 1:
        log.info(
            "Reprise du scénario %d à la partie %d/%d.",
            scenario_id, start_game, NUM_GAMES,
        )

    log.info(
        "=== Scénario %d | X=%s vs Y=%s | Historique=%s | %d parties restantes ===",
        scenario_id, x_model, y_model, has_history, remaining,
    )

    # Reconstruction du log en mémoire (pour le brouillard de guerre sur reprise)
    game_log: list[dict] = existing.to_dict("records") if not existing.empty else []

    # --- Boucle principale ---
    for game_num in range(start_game, start_game + remaining):

        x_card, y_card = deal_cards()

        # Historique Markdown filtré par joueur (brouillard de guerre)
        history_x_md = build_history_table_for_x(game_log) if has_history else ""
        history_y_md = build_history_table_for_y(game_log) if has_history else ""

        # Action de X (Abattre=1 ou Miser=2)
        x_action, x_fallback = call_llm(
            model           = x_model,
            system_prompt   = SYSTEM_PROMPT_X,
            user_prompt     = build_user_prompt_x(x_card, history_x_md),
            valid_actions   = {ACTION_ABATTRE, ACTION_MISER},
            fallback_action = ACTION_ABATTRE,
            role_label      = "X",
            game_id         = game_num,
        )

        # Action de Y (uniquement si X a Misé)
        y_action:   Optional[int] = None
        y_fallback: bool          = False

        if x_action == ACTION_MISER:
            y_action, y_fallback = call_llm(
                model           = y_model,
                system_prompt   = SYSTEM_PROMPT_Y,
                user_prompt     = build_user_prompt_y(y_card, history_y_md),
                valid_actions   = {ACTION_PASSER, ACTION_SUIVRE},
                fallback_action = ACTION_PASSER,
                role_label      = "Y",
                game_id         = game_num,
            )

        # Calcul des gains nets
        winner, x_payoff, y_payoff = compute_payoffs(x_card, y_card, x_action, y_action)
        is_fallback = x_fallback or y_fallback

        # Log console structuré
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

        # Mise à jour du log en mémoire (pour l'historique des prochains tours)
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

        # Écriture immédiate dans le CSV (mode append — résistant aux interruptions)
        pd.DataFrame([{
            "Game_ID":      game_num,
            "X_Model":      x_model,
            "Y_Model":      y_model,
            "Has_History":  has_history,
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

    log.info(
        "=== Scénario %d terminé. %d parties enregistrées → %s ===",
        scenario_id, NUM_GAMES, csv_path,
    )


# ==============================================================================
# 7. POINT D'ENTRÉE CLI — argparse
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Kuhn Poker — Évaluation de la rationalité des LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Scénarios disponibles :
  1  Mistral(X) vs OpenAI(Y)  | sans historique   [has_history=False]
  2  Mistral(X) vs OpenAI(Y)  | avec historique   [has_history=True]
  3  OpenAI(X)  vs Mistral(Y) | sans historique   [has_history=False]
  4  OpenAI(X)  vs Mistral(Y) | avec historique   [has_history=True]

Lancement en parallèle (4 terminaux) :
  python main.py --scenario 1
  python main.py --scenario 2
  python main.py --scenario 3
  python main.py --scenario 4
""",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        metavar="[1-4]",
        help="Numéro du scénario à exécuter",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    scenario_id            = args.scenario
    x_model, y_model, has_history = SCENARIOS[scenario_id]

    log.info(
        "Lancement — scénario %d | X=%s | Y=%s | Historique=%s",
        scenario_id, x_model, y_model, has_history,
    )

    run_scenario(
        scenario_id = scenario_id,
        x_model     = x_model,
        y_model     = y_model,
        has_history = has_history,
    )