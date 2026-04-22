# Miryang Traffic Accident Risk Prediction
### XAI + DiCE + Lift Chart + LLM-based Policy Report Framework

> **Paper:** 읍면동 단위 교통사고 위험지수 예측을 위한 XAI 및 반사실적 분석 기반 정책 지원 프레임워크  
> **Journal:** 
> **DOI:** 

---

## Overview

This repository contains the full analysis pipeline for predicting traffic accident risk at the **읍면동 (sub-district) level** in Miryang-si, South Gyeongsang Province, South Korea.

The framework integrates:
- **LightGBM** regression and classification models for risk index prediction
- **SHAP** for global and local explainability
- **DiCE** counterfactual analysis for actionable policy recommendations
- **Lift Chart** for resource allocation and over-warning detection
- **GPT-4o-mini + RAG** for automated natural-language policy reports
- **RAGAS** for automated quality evaluation of generated reports
- **Sensitivity Analysis** for robustness validation of the expert-survey-based risk index

```
Raw Data (CSV)
    │
    ▼
Feature Engineering ──► LightGBM Regressor  ──► SHAP (Why risky?)
                    └──► LightGBM Classifier ──► Lift Chart (Resource allocation?)
                                                ──► DiCE (What to change?)
                                                         │
                                                         ▼
                                               LLM Report Agent (GPT-4o-mini + RAG)
                                                         │
                                                         ▼
                                               RAGAS Evaluation (Report quality)
```

---

## Repository Structure

```
├── 1_miryang_analysis.py              # Step 1: Modeling + XAI + Comparison + Figures
├── 2_miryang_report_agent.py          # Step 2: GPT-based policy report agent
├── 3_miryang_sensitivity_analysis.py  # Step 3: Weight sensitivity analysis
├── 4_miryang_ragas_evaluation.py      # Step 4: RAGAS-based report quality evaluation
├── .env.example                       # API key template
├── README.md
│
├── outputs/
│   ├── figures/
│   │   ├── Fig1_Confusion_Matrix.png
│   │   ├── Fig2_SHAP_Bar.png
│   │   ├── Fig3_SHAP_Beeswarm.png
│   │   ├── Fig4_Lift_Chart.png
│   │   ├── Fig5_DiCE_Policy.png
│   │   ├── Fig6_Pipeline.png
│   │   ├── Fig7_Regression_Comparison.png
│   │   ├── Fig8_Classification_Comparison.png
│   │   ├── Fig9_CV_AUC_ErrorBar.png
│   │   ├── Fig10_Radar_Chart.png
│   │   ├── Fig11_Sensitivity_AUC.png       # Sensitivity analysis results
│   │   └── Fig12_Prompt_Report_Example.png # Prompt structure + report example
│   │
│   ├── tables/
│   │   ├── TableI_Regression_Comparison.csv
│   │   ├── TableII_Classification_Comparison.csv
│   │   ├── TableIII_Lift_Chart.csv
│   │   ├── TableIV_Alert_Classification.csv
│   │   ├── TableV_DiCE_Policy.csv
│   │   ├── TableVI_LLM_Dataset.csv
│   │   ├── TableVII_Report_Summary.csv
│   │   ├── TableVIF_Diagnosis.csv          # VIF multicollinearity diagnosis
│   │   ├── TableS1_Sensitivity_AUC.csv     # Sensitivity: AUC by scenario
│   │   ├── TableS2_Sensitivity_RankOverlap.csv  # Sensitivity: rank stability
│   │   ├── TableS3_Sensitivity_Correlation.csv  # Sensitivity: Spearman r
│   │   ├── TableR1_RAGAS_Per_Sample.csv    # RAGAS: per-district scores
│   │   └── TableR2_RAGAS_Summary.csv       # RAGAS: summary statistics
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
| Unit | 읍면동 (sub-district) × month |
| Observations | 447 |
| Features | 38 independent variables (5 categories) |
| Target | Traffic Risk Index (expert-survey weighted accident severity score) |

**Feature Categories**

| Category | Variables | Count |
|---|---|---|
| Traffic violations | viol_unsafe_driving, viol_signal, viol_pedestrian_prot, ... | 11 |
| Road surface | road_dry, road_wet, road_frost, road_snow, ... | 5 |
| Weather | weather_clear, weather_cloudy, weather_rain, ... | 6 |
| Road type | road_intersection, road_near_intersection, ... | 7 |
| Regional characteristics | population, transport_companies, tourist_spots, ... | 9 |

**VIF Multicollinearity**

VIF iterative elimination was applied to linear models only. Three variables (`viol_unsafe_driving`, `road_dry`, `weather_clear`) were identified with VIF=∞ and excluded from linear regression and logistic regression training. Tree-based ensemble models and MLP used all 33 features.

> **Note:** Raw data is not included due to redistribution restrictions.  
> Please download from [TAAS](https://taas.koroad.or.kr) and Miryang-si open data portal.

---

## Installation

```bash
# Core
pip install lightgbm xgboost catboost shap scikit-learn statsmodels

# XAI
pip install dice-ml

# LLM agent
pip install openai python-dotenv

# RAGAS evaluation
pip install ragas langchain langchain-openai datasets

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
python 1_miryang_analysis.ipynb
```

**Outputs:** Figures (Fig1–Fig10, PNG) + Tables I–VI (CSV) + VIF diagnosis table

Estimated runtime: 3–10 minutes (SHAP computation is the bottleneck).

### Step 3. Run LLM report agent

```bash
python 2_miryang_report_agent.ipynb
```

**Outputs:** `traffic_reports.json`, `traffic_reports.txt`, `TableVII_Report_Summary.csv`

### Step 4. Run sensitivity analysis

```bash
python 3_miryang_sensitivity_analysis.ipynb
```

**Outputs:** TableS1–S3 (CSV) + Fig11 (PNG)  
Tests 7 weight perturbation scenarios (expert_mean ±10%, ±20%, ±30%).

### Step 5. Run RAGAS evaluation

```bash
python 4_miryang_ragas_evaluation.ipynb
```

**Outputs:** TableR1–R2 (CSV)  
Evaluates generated reports on Faithfulness, Answer Relevancy, Context Recall, Context Precision.

---

## Key Results

### Classification Performance (LightGBM selected)

| Model | Accuracy | F1-Score | ROC-AUC | CV AUC | Time (s) | Features Used |
|---|---|---|---|---|---|---|
| **LightGBM (Ours)** | **0.9000** | **0.8966** | 0.9689 | 0.9452 | **0.02** | 33 (all) |
| CatBoost | 0.9000 | 0.8966 | 0.9704 | 0.9476 | 0.20 | 33 (all) |
| Logistic Reg. | 0.8889 | 0.8837 | **0.9733** | 0.9498 | 2.60 | 30 (VIF-filtered) |
| XGBoost | 0.8889 | 0.8889 | 0.9713 | 0.9453 | 0.05 | 33 (all) |
| Random Forest | 0.8889 | 0.8837 | 0.9649 | 0.9492 | 0.16 | 33 (all) |
| MLP | 0.8333 | 0.8387 | 0.9130 | 0.9208 | 0.70 | 33 (all) |

LightGBM achieves equivalent accuracy to CatBoost while being **15× faster**, making it optimal for real-time service deployment.

### Regression Performance

| Model | R² | RMSE | MAE | CV R² | Features Used |
|---|---|---|---|---|---|
| Lasso Reg. | **0.8664** | 15.2278 | 10.7599 | 0.8533 | 30 (VIF-filtered) |
| Ridge Reg. | 0.8665 | 15.2260 | 11.3837 | 0.8533 | 30 (VIF-filtered) |
| XGBoost | 0.8205 | 17.6525 | 12.5359 | 0.8277 | 33 (all) |
| Random Forest | 0.8177 | 17.7890 | 12.5781 | 0.8210 | 33 (all) |
| LightGBM | 0.8154 | 17.9005 | 12.3388 | 0.8281 | 33 (all) |
| CatBoost | 0.8113 | 18.1003 | 12.4073 | 0.8316 | 33 (all) |
| MLP | 0.7333 | 21.5199 | 14.6364 | 0.7183 | 33 (all) |
| Linear Reg. | −6.40×10¹⁹ | — | — | — | 30 (VIF-filtered) |

### Top SHAP Risk Factors

1. `road_dry` — dry road surface incidents (Mean |SHAP| ≈ 19.0)
2. `viol_unsafe_driving` — unsafe driving violations (≈ 5.1)
3. `road_wet` — wet road incidents (≈ 3.0)

> `road_dry` reflects statistical correlation with accident exposure frequency, not direct causation. It is treated as an **immutable feature** in DiCE analysis. Policy focus should target **enforcement intensity** targeting law violations.

### Lift Chart

- Top 10% selection → Decile Lift **1.957** (1.96× better than random)
- Top 30% coverage → captures **58.7%** of all actual high-risk districts
- Provides quantitative basis for budget-constrained traffic safety resource allocation

### Sensitivity Analysis (Weight Robustness)

7 scenarios perturbing `expert_mean` by ±10%, ±20%, ±30% (preset_w fixed):

| Metric | Result |
|---|---|
| ROC-AUC range | 0.9066 – 0.9635 (range: 0.057) |
| Top-10 rank overlap vs baseline | 100% across all scenarios |
| Min Spearman r (TRI ranking) | 0.9961 |
| Baseline TRI vs original Spearman r | 0.9198 |

Confirms robustness of the risk index under expert weight uncertainty.

### RAGAS Report Quality Evaluation

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Faithfulness | 0.2436 | 0.0708 | 0.1579 | 0.3182 |
| Answer Relevancy | **0.9449** | 0.0067 | 0.9342 | 0.9509 |
| Context Recall | 0.6000 | 0.1491 | 0.3333 | 0.6667 |
| Context Precision | 0.2105 | 0.1859 | 0.0734 | 0.5167 |

High Answer Relevancy (0.94) confirms strong query-response alignment. Lower Faithfulness reflects the numeric-centric RAG context; improvable via Advanced RAG techniques.

### LLM Report Sample (삼문동, 2021-07)

```
District  : 삼문동 | Period: 2021-07 | Risk Index: 192.40 | Alert: High-Risk Warning

[EXECUTIVE SUMMARY]
The traffic accident risk index stands at 192.40, categorizing it as high-risk.
Primary drivers include dry road surface incidents and unsafe driving violations.

[TOP POLICY RECOMMENDATIONS]
[1] Targeted enforcement on unsafe driving violations — Short-term (1-3 months)
[2] Public awareness campaign on road surface risks — Mid-term (3-12 months)
[3] Pedestrian protection at intersections — Long-term (1+ year)

[MONITORING KPIs]
• Reduce dry road surface incidents by 20% within 6 months
• Decrease unsafe driving violations by 15% within 3 months
• Increase pedestrian compliance by 25% within 12 months
```

---

## Methodology

### Risk Index Design

The traffic risk index is defined as:

```
TRI_i = Σ (n_ik × w_k)   for k = 1, ..., 12 accident types
```

where:
- `n_ik` = count of accident type `k` in district `i`
- `w_k = expert_mean_k + preset_w_k` (final weight)
- `expert_mean`: average severity score from 3 experts (0–10 scale)
- `preset_w`: pre-assigned weight reflecting social impact by severity class
  - Fatal: 50, Serious: 20–30, Minor: 3–6, Reported injury: 3–10

### Binary Classification Threshold

```
y_cls = 1  if TRI >= median(TRI)
y_cls = 0  otherwise
```

### Hyperparameter Settings

| Model | Key Parameters |
|---|---|
| LightGBM | n_estimators=100, max_depth=2 |
| XGBoost | n_estimators=100, max_depth=3, lr=0.1 |
| CatBoost | iterations=100, depth=4, lr=0.1 |
| Random Forest | n_estimators=100, max_depth=6 |
| MLP | hidden=(128,64,32), max_iter=500 |
| Logistic Reg. | max_iter=1000 |
| Ridge | α=1.0 |
| Lasso | α=0.1, max_iter=5000 |

No automated hyperparameter search (GridSearch/Optuna) was performed; conservative defaults were chosen to prevent overfitting on the small dataset (n=447), validated via 5-fold cross-validation.

### XAI Pipeline

```
SHAP  → "Why is this district high-risk?"     (global + local feature attribution)
DiCE  → "What needs to change?"               (counterfactual policy scenarios)
        actionable: viol_unsafe_driving, viol_signal, road_intersection
        immutable:  road_dry, weather_clear (environmental variables)
Lift  → "How efficiently can we allocate resources?"  (decile-based prioritization)
```

---

## Citation

If you use this code or methodology, please cite:

```
@article{miryang2026,

```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- 도로교통공단 교통사고분석시스템 (TAAS) for traffic accident data
- Miryang-si for regional public data
