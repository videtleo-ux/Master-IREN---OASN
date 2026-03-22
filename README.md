# Rationalité Stratégique des LLMs en Environnement à Information Imparfaite

Code et données associés à l'article *"Rationalité Stratégique des LLM en Environnement à Information Imparfaite"* (Videt, 2026, Master IREN).

## Structure du dépôt

```
├── main.py              # Expérience 1 — Confrontation inter-modèles (2 000 parties)
├── main_exp2.py         # Expérience 2 — Self-play gpt-5.4-mini (1 500 parties)
├── main_exp3.py         # Expérience 3 — Effet des personas (4 000 parties)
├── analyze_v2.py        # Analyse exp1 et exp2 (KPIs, régressions, exploitation adaptative)
├── analyze_v3.py        # Analyse exp3 (KPIs, comparaison baseline, régressions persona)
├── requirements.txt
├── results/
│   └── report_summary.csv      # KPIs exp1
├── results_exp2/
│   └── report_summary.csv      # KPIs exp2
└── results_exp3/
    └── report_summary.csv      # KPIs exp3
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` à la racine :

```
MISTRAL_API_KEY=your_mistral_key
OPENAI_API_KEY=your_openai_key
```

## Lancer les expériences

Les scénarios peuvent être lancés en parallèle (un terminal par scénario).

**Expérience 1** (scénarios 1 à 4) :
```bash
python main.py --scenario 1
python main.py --scenario 2
python main.py --scenario 3
python main.py --scenario 4
```

**Expérience 2** (scénarios 1 à 3) :
```bash
python main_exp2.py --scenario 1
python main_exp2.py --scenario 2
python main_exp2.py --scenario 3
```

**Expérience 3** (scénarios 1 à 8) :
```bash
python main_exp3.py --scenario 1
# ... jusqu'à --scenario 8
```

## Analyser les résultats

```bash
python analyze_v2.py --dir results          # Expérience 1
python analyze_v2.py --dir results_exp2     # Expérience 2
python analyze_v3.py --exp3 results_exp3 --baseline results  # Expérience 3
```

Les figures sont exportées dans `results/figures/`, `results_exp2/figures/`, `results_exp3/figures/`.

## Données disponibles

Les fichiers `report_summary.csv` dans chaque dossier contiennent les KPIs agrégés par scénario (bluff rate, call rate, gains nets, fallbacks). Les données brutes partie par partie (`scenario_N.csv`) ne sont pas incluses dans ce dépôt en raison de leur taille mais sont reproductibles en relançant les scripts.

## Modèles utilisés

- `mistral/mistral-small-latest` (Mistral AI)
- `openai/gpt-5.4-mini` (OpenAI)

## Citation

```
Videt, L. (2026). Rationalité Stratégique des LLM en Environnement à Information Imparfaite.
Master IREN. 
```
