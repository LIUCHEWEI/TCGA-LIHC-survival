# TCGA-LIHC-survival

Survival analysis on TCGA liver hepatocellular carcinoma (LIHC) data, combining
mutation-derived features (TMB) with machine learning survival models.

## Contents

- `Dataset/` — merged dataset (clinical data + TMB) used for modeling
- `Mutations_derived_features_calculation/` — scripts for deriving mutation-based
  features (e.g. `pTMB_calculation.R`)
- `XGBoost_survival.py` — XGBoost-based survival model with C-index evaluation
  and SHAP-based feature importance
- `MLP_DeepSurv.py` — MLP DeepSurv model for survival prediction

## Requirements

- Python 3.x with `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`,
  `lifelines`, `shap`, `matplotlib`, `seaborn`
- R with `maftools`, `dplyr`, `tidyr`, `ggplot2`, `GenomicRanges`, `facets`
  (for `pTMB_calculation.R`)
