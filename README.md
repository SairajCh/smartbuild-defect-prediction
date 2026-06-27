# SmartBuild Defect Prediction

Predicting manufacturing defects before they occur using machine learning — 
binary and multiclass classification on pre-production sensor data.

## Results at a Glance

| Task | Best Model | Accuracy |
|---|---|---|
| Defect Detection (binary) | Random Forest / Decision Tree | **92.2%** |
| Defect Type (multiclass) | Random Forest | ~52% (data ceiling) |

**Key finding:** Ionization Class alone accounts for 68% of predictive power.  
A depth-2 Decision Tree matches a 100-tree Random Forest — the boundary is simple.

## Business Context

- Defect caught **before** machine → costs €10 (discard raw material)  
- Defect caught **after** machine → costs €150 (machine time + waste)  
- Model achieves **97% recall** on defective products

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn` · `XGBoost`

## Project Structure
notebooks/   → full analysis notebook (EDA → modelling → business impact)
data/        → production log + machine settings CSV files
outputs/     → presentation slides + exported figures

## Methods

- Exploratory Data Analysis (ionization class crosstab, boxplots, correlation)
- Binary classification: Decision Tree (depth-tuned), Random Forest, Logistic Regression
- Multiclass classification: DT, RF, LR, Gradient Boosting, XGBoost, KNN
- PCA visualisation of class separability
- Cost-based business impact analysis (confusion matrix → €)

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter
jupyter notebook notebooks/SmartBuild_Defect_Prediction.ipynb
```
> Data files required: `Production_Log_01.csv` and `Machine_Settings_Log_01.csv`  
> Update `DATA_DIR` at the top of the notebook if files are in a different folder.
