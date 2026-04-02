# Miryang Traffic Accident Risk Prediction
### XAI + DiCE + Lift Chart + LLM-based Policy Report Framework

> **Paper:** Predicting Traffic Accident Risk Index in Miryang-si Using Explainable AI and LLM-based Automated Policy Reports  
> **Journal:**  
> **Authors:** 

---

## Overview

This repository contains the full analysis pipeline for predicting traffic accident risk at the **읍면동 (sub-district) level** in Miryang-si, South Gyeongsang Province, South Korea.

The framework integrates:
- **LightGBM** regression and classification models for risk index prediction
- **SHAP** for global and local explainability
- **DiCE** counterfactual analysis for actionable policy recommendations
- **Lift Chart** for model reliability and over-warning detection
- **GPT-4o-mini + RAG** for automated natural-language policy reports

```
Raw Data (CSV)
    │
    ▼
Feature Engineering ──► LightGBM Regressor  ──► SHAP (Why risky?)
                    └──► LightGBM Classifier ──► Lift Chart (Over-warning?)
                                                ──► DiCE (What to change?)
                                                         │
                                                         ▼
                                               LLM Chatbot (Policy Report)
```

---

## Repository Structure

```
├── 1_miryang_analysis.py              # Step 1: Modeling + XAI + Comparison + Figures
├── 2_miryang_report_agent.py          # Step 2: GPT-based policy report agent
├── .env.example                       # API key template
├── README.md
│
├── outputs/
│   ├── figures/
│   │   ├── fig1.png
│   │   ├── fig2.png
│   │   ├── fig3.png
│   │   ├── fig4.png
│   │   ├── fig5.png
│   │   ├── fig6.png
│   │   ├── fig7.png
│   │   ├── fig8.png
│   │   ├── fig9.png
│   │   └── fig10.png
│   │
│   ├── tables/
│   │   ├── TableI_Regression_Comparison.csv
│   │   ├── TableII_Classification_Comparison.csv
│   │   ├── TableIII_Lift_Chart.csv
│   │   ├── TableIV_Alert_Classification.csv
│   │   ├── TableV_DiCE_Policy.csv
│   │   ├── TableVI_LLM_Dataset.csv
│   │   └── TableVII_Report_Summary.csv
│   │
│   ├── traffic_reports.json           # RAG knowledge base
│   └── traffic_reports.txt            # Human-readable policy reports
```

---

## Dataset

| Item | Detail |
|---|---|
| Source | 도로교통공단 교통사고분석시스템 (TAAS) + Miryang-si public data |
| Period | 2020 – 2022 |
| Unit | 읍면동 (sub-district) × quarter |
| Observations | 447 |
| Features | 38 independent variables (5 categories) |
| Target | Traffic Risk Index (weighted accident severity score) |

**Feature Categories**

| Category | Variables | Count |
|---|---|---|
| Traffic violations | viol_unsafe_driving, viol_signal, viol_pedestrian_prot, ... | 11 |
| Road surface | road_dry, road_wet, road_frost, road_snow, ... | 5 |
| Weather | weather_clear, weather_cloudy, weather_rain, ... | 6 |
| Road type | road_intersection, road_near_intersection, ... | 7 |
| Regional characteristics | population, transport_companies, tourist_spots, ... | 9 |

> **Note:** Raw data is not included in this repository due to redistribution restrictions.  
> Please download from [TAAS](https://taas.koroad.or.kr) and Miryang-si open data portal.

---

## Installation

```bash
# Core
pip install lightgbm xgboost catboost shap scikit-learn

# XAI
pip install dice-ml

# LLM agent
pip install openai python-dotenv

# Optional (TabNet — skip if DLL errors occur on Windows)
pip install pytorch-tabnet
```

**Python version:** 3.8 or higher recommended

---

## Quick Start

### Step 1. Set up API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
```

### Step 2. Run modeling pipeline

```bash
python 1_miryang_analysis.py
```

**Outputs:** All figures (PNG) + Tables I–VI (CSV)

Estimated runtime: 3–10 minutes depending on hardware (SHAP computation is the bottleneck).

### Step 3. Run LLM report agent

```bash
python 2_miryang_report_agent.py
```

**Outputs:** `traffic_reports.json`, `traffic_reports.txt`, `TableVII_Report_Summary.csv`

---

## Key Results

### Classification Performance (LightGBM selected)

| Model | Accuracy | F1-Score | ROC-AUC | CV AUC | Time (s) |
|---|---|---|---|---|---|
| **LightGBM (Ours)** | **0.9000** | **0.8966** | 0.9689 | 0.9452 | **0.02** |
| CatBoost | 0.9000 | 0.8966 | 0.9704 | 0.9476 | 0.20 |
| Logistic Reg. | 0.8889 | 0.8837 | **0.9733** | 0.9498 | 2.60 |
| XGBoost | 0.8889 | 0.8889 | 0.9713 | 0.9453 | 0.05 |
| Random Forest | 0.8889 | 0.8837 | 0.9649 | 0.9492 | 0.16 |
| MLP | 0.8333 | 0.8387 | 0.9130 | 0.9208 | 0.70 |

LightGBM achieves equivalent accuracy to CatBoost while being **15× faster**, making it optimal for real-time service deployment.

### Top SHAP Risk Factors

1. `road_dry` — dry road surface incidents (Mean |SHAP| ≈ 19.0)
2. `viol_unsafe_driving` — unsafe driving violations (≈ 5.1)
3. `road_wet` — wet road incidents (≈ 3.0)

> `road_dry` is an environmental variable that cannot be directly reduced. Policy focus should target **enforcement intensity under dry road conditions**.

### Lift Chart

- Top 10% selection → Decile Lift **1.957** (1.96× better than random)
- Top 30% coverage → captures **58.7%** of all actual high-risk districts
- Provides quantitative basis for budget-constrained traffic safety resource allocation

### LLM Report Sample (삼문동, 2021-07)

```
District  : 삼문동 | Period: 2021-07 | Risk Index: 192.40 | Alert: High-Risk Warning

[EXECUTIVE SUMMARY]
The traffic accident risk index stands at 192.40, categorizing it as high-risk.
Primary drivers include dry road surface incidents and unsafe driving violations.

[TOP POLICY RECOMMENDATION]
Implement targeted enforcement on unsafe driving violations (Short-term: 1-3 months)

[MONITORING KPI]
Reduce dry road surface incidents by 20% within 6 months
```

---

## Methodology

### Risk Index Design

The traffic risk index is defined as:

```
traffic_risk_index_i = Σ (n_ik × w_k)   for k = 1, ..., 12 accident types
```

where `n_ik` is the count of accident type `k` in district `i`, and `w_k` is the severity weight derived from expert survey (scale 0–10) combined with pre-assigned accident-type weights.

### Binary Classification Threshold

```
y_cls = 1  if traffic_risk_index >= median
y_cls = 0  otherwise
```

### XAI Pipeline

```
SHAP  → "Why is this district high-risk?"   (global + local feature attribution)
DiCE  → "What needs to change?"             (counterfactual policy scenarios)
Lift  → "How reliable is the alert?"        (over-warning detection by decile)
```

---

## Citation

If you use this code or methodology, please cite:

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- 도로교통공단 교통사고분석시스템 (TAAS) for traffic accident data
- Miryang-si for regional public data
