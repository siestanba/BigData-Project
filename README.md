# Minimal 32K Repro (Notebook-Only)

This package is intentionally simple:

```text
projet/
├── notebooks/
│   ├── 01_train_eval_32k_from_mongo.ipynb
│   └── 02_predict_custom_comments.ipynb
├── weights/                # optional pretrained weights
├── outputs/                # generated artifacts
├── requirements.txt
└── README.md
```

No external Python scripts are required.

## Prerequisites

- Python 3.10+
- MongoDB available (default: `mongodb://localhost:27017`)
- A normalized collection in MongoDB (default: `bd_team_normalized.normalized_reviews`)

Expected useful fields in MongoDB documents:

- `text` (string)
- `label_binary` in `{0,1}`
- `source` in `{twitter, amazon_reviews, social_mixed, steam_ubisoft}`
- `created_at` (optional but used for time plots)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Notebook 1: Train + Evaluate 32K

Open and run all cells in:

- `notebooks/01_train_eval_32k_from_mongo.ipynb`

At the top, choose:

- `THEME = "twitter" | "amazon_reviews" | "social_mixed" | "steam_ubisoft" | "all_sources"`

Default protocol is already set to 32K:

- `TRAIN_SIZE = 32000`
- `EVAL_SIZE = 6400`

Behavior:

- if `weights/<theme>_distilroberta/hf_model` exists and `USE_PRETRAINED_IF_EXISTS=True`, it loads the model
- otherwise it trains from MongoDB and saves weights

Outputs generated automatically:

- `weights/<theme>_distilroberta/hf_model/`
- `outputs/<theme>/metrics_summary.json`
- `outputs/<theme>/training_metrics.json`
- figures (confusion matrix, ROC, PR, trends, hashtags, wordclouds)

## Notebook 2: Predict New Comments

Open and run all cells in:

- `notebooks/02_predict_custom_comments.ipynb`

Set:

- `THEME` (typically `all_sources`)
- the `comments` list directly in the notebook

Outputs:

- `outputs/<theme>/custom_comments_predictions.csv`
- `outputs/<theme>/custom_comments_predictions.json`

## Notes

- This is a notebook-only package by design.
- If you want to publish pretrained weights, place them under `weights/` with the same folder naming convention.
