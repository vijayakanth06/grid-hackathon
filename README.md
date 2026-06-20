# ParkSense AI — Spatiotemporal Parking Violation & Incident Intelligence

This repository contains the complete implementation for **Flipkart Gridlock Hackathon 2.0 - Round 2 (Prototype Phase) - Theme 1 (Poor Visibility on Parking-Induced Congestion)**, cross-referenced with **Theme 2 (Event-Driven Congestion)** ASTraM events.

---

## 📂 Folder Structure

```
d:\grid\
├── dataset\
│   ├── jan to may police violation_anonymized791b166.csv                       # Theme 1 (298K records)
│   └── Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv  # Theme 2 (8.1K records)
├── src\
│   ├── config.py             # Dual-environment configuration & hyperparams
│   ├── checkpoint.py         # Checkpoint & resume system for fault-tolerant training
│   ├── gpu_utils.py          # GPU-aware model factory for XGBoost, LightGBM, and CatBoost
│   ├── hpo.py                # HPO optimization using Optuna
│   ├── data_loader.py        # Dataset ingestion & preprocessing
│   ├── feature_engineering.py # Spatiotemporal grid cell generation, cyclical & lag features
│   ├── cross_reference.py    # Theme 2 incident cross-referencing features
│   ├── model_trainer.py      # Multi-model training pipeline with Stacking & TabM
│   ├── evaluator.py          # Leaderboard builder & evaluation metrics
│   ├── forecaster.py         # Prophet daily forecaster (Model D)
│   └── utils.py              # Haversine distance & SHAP explainability utils
├── app.py                    # Streamlit interactive dashboard
├── run_pipeline.py            # Master pipeline script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Environment Setup

Activate the `data-env` conda environment:
```bash
conda activate data-env
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Verification (Smoke Test)
To verify that all packages are imported correctly and can access GPU acceleration:
```bash
python smoke_test.py
```

---

## 🚀 Running the Training Pipeline

The training pipeline runs HPO, performs feature engineering, and trains **14+ machine learning models** (including XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, TabM deep learning, and a Stacked Ensemble) for both classification and regression tasks. 

It is designed with **GPU auto-detection** (will leverage Tesla V100 GPU on the server and fallback to CPU on Windows) and **fault tolerance with checkpointing** (if interrupted, running it again will automatically resume from the last completed model).

To execute the training pipeline:
```bash
python run_pipeline.py
```

---

## 🖥️ Running the Streamlit Dashboard

To launch the interactive dashboard locally:
```bash
streamlit run app.py
```

---

## 📦 Submission Package Checklist

Before submitting, package the folder by compressing:
1. All python scripts (excluding the raw datasets or heavy model joblib files if size exceeds 100MB).
2. The Streamlit dashboard link (`https://parksense-ai.streamlit.app`).
3. The demo screen recording video.
