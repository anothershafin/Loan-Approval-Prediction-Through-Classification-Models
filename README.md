# Loan Approval Prediction Project

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## Overview
This repository contains a complete, end-to-end workflow for **loan approval prediction** as part of BRACU **CSE422** coursework (Group 04). The notebook performs data loading, EDA, cleaning, encoding, scaling, supervised modeling, metrics visualization, and an **unsupervised** KMeans baseline for reference.

## Dataset
- **Source**: Downloaded via `gdown` from Google Drive using a hardcoded `file_id` inside the notebook.
- **Shape mentioned in notebook**: ~45,000 rows × ~14 columns.
- **Target**: `loan_status` (binary).
- **Imbalance noted**: ≈ 78% vs 22% (Not Approved vs Approved), as described in the notebook text.

## Project Workflow

1) **Exploratory Data Analysis (EDA)**
   - Checked class distribution of `loan_status`.
   - Separated **quantitative** (numeric) and **categorical** features.
   - Ran descriptive statistics for numeric and categorical columns.
   - Visualized group-wise means for categorical columns vs target.
   - Computed and visualized correlations (Pearson / Spearman / Kendall) for numeric features against `loan_status`.

2) **Missing Values**
   - Built a per-column table of missing counts and percentages.
   - **Dropped columns** whose missing percentage exceeded a threshold (40% in notebook).
   - If **rows with any missing values** accounted for **< 5%** of the dataset, they were dropped; otherwise imputations were applied:
     - **Numeric**: mean imputation (`SimpleImputer(strategy="mean")`).
     - **Categorical**: mode/most-frequent imputation (`SimpleImputer(strategy="most_frequent")`).
   - Plotted a post-processing missing-values heatmap for a final check.

3) **Encoding**
   - One-Hot Encoding for categorical variables using `pd.get_dummies(..., drop_first=True)`. 

4) **Train/Test Split & Scaling**
   - `train_test_split` with **80/20** hold-out and `random_state=42`.
   - Built a list of numeric columns and **kept binary/dummy columns unscaled**.
   - Standardized the remaining continuous columns with `StandardScaler` (fit on train, transform on test).

5) **Supervised Models**
   - **Logistic Regression** (`LogisticRegression(max_iter=1000, random_state=42)`)
   - **Decision Tree** (`DecisionTreeClassifier(random_state=42, ...)`)
   - **Neural Network (MLPClassifier)** with ReLU/Adam and early stopping.
   - For each: predicted labels and probabilities, computed metrics, confusion matrix heatmap, and ROC curve.

6) **Metrics & Comparison**
   - Computed **Accuracy, Precision, Recall, F1, ROC-AUC** via a helper `log_and_store_metrics(...)`.
   - Aggregated results into a comparison DataFrame and plotted bar charts for Accuracy and for Precision/Recall/F1/AUC.

7) **Unsupervised (Bonus)**
   - **KMeans (k=2)** trained on scaled train features as an unsupervised baseline.
   - Cluster IDs were **aligned** to ground truth labels to report reference Accuracy/Precision/Recall/F1.
   - Reported **Silhouette Score** (on train) and a confusion matrix vs. true labels (for reference only).


## Results (at a glance)

The notebook prints per-model metrics and shows:
- Confusion matrices for Logistic Regression, Decision Tree, and MLP.
- ROC curves with AUC values.
- A comparison table and bar charts for accuracy, precision, recall, F1, and AUC.
> Exact scores depend on the dataset you download and may vary across runs if you change random seeds or parameters.


## Environment & Dependencies

- Python 3.x
- Core libraries used (as inferred from the notebook):
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn`
  - `scikit-learn` (model_selection, preprocessing, impute, metrics, linear_model, tree, neural_network, cluster, decomposition)
  - `gdown` (for dataset download)


## How to Run

- Open `Loan Approval Project.ipynb` in Jupyter/Colab.
- Ensure `gdown` is installed (the notebook installs it with `!pip install gdown`).
- **Dataset**: The notebook calls `gdown.download(...)` with a predefined Google Drive `file_id` and saves a CSV as `Loan Approval Dataset.csv`.
  - If the link/file_id stops working, replace the `file_id` in the notebook with your own and re-run the first cells.
- Run all cells in order (top to bottom).


## Reproducibility Notes

- **Reproducibility**: `random_state=42` is used in several steps (train/test split, models). For perfectly repeatable results, keep these seeds constant and run the notebook deterministically.


## Repository Structure

```
.
├── Loan Approval Project.ipynb
├── README.md  ← you are here
└── (auto-downloaded) Loan Approval Dataset.csv  # created when you run the notebook
```


## Next Steps / Ideas

- Hyperparameter tuning (e.g., GridSearchCV/RandomizedSearchCV).
- Add cross-validation and calibration (e.g., Platt scaling) for better probability estimates.
- Try class imbalance strategies (e.g., class weights, resampling/SMOTE).
- Add model explainability (feature importance, SHAP) to interpret predictions.
- Package the pipeline (preprocessing + model) with `Pipeline` and persist with `joblib`.


## Acknowledgements
- Course: **CSE422** — Machine Learning / AI
- Libraries: **NumPy**, **Pandas**, **scikit-learn**, **Matplotlib**, **Seaborn**
- Data hosted on Google Drive and fetched via **gdown** from within the notebook.
