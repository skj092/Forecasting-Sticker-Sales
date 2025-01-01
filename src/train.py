import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

# Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ----------------- CONFIG -------------------
# You can tune these or load from a hyperparameter search
xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.0,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'random_state': 42
}

lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': -1,
    'num_leaves': 64,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'verbosity': -1,
    'random_state': 42
}

cat_params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 8,
    'loss_function': 'MAPE',
    'random_state': 42,
    'silent': True
}

N_SPLITS = 5  # KFold splits

# ----------------- READ DATA -------------------
# Kaggle environment:
#   /kaggle/input/playground-series-s5e1/train.csv
#   /kaggle/input/playground-series-s5e1/test.csv
#   /kaggle/input/playground-series-s5e1/sample_submission.csv
path = Path("/home/sonujha/rnd/Forecasting-Sticker-Sales/data/")
train_data = pd.read_csv(path/'train.csv')
test_data  = pd.read_csv(path/'test.csv')
sample_sub = pd.read_csv(path/'sample_submission.csv')

print("Initial train_data shape:", train_data.shape)
print("Initial test_data shape: ", test_data.shape)

# ----------------- CLEAN & PREPARE TRAIN -------------------
# 1) Drop duplicates
train_data.drop_duplicates(inplace=True)

# 2) Drop missing target rows
train_data.dropna(subset=['num_sold'], inplace=True)

# 3) Convert 'date' to datetime if needed
train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
test_data['date']  = pd.to_datetime(test_data['date'],  errors='coerce')

# 4) Extract date features (you can add day-of-week, etc. if you want)
train_data['Year']  = train_data['date'].dt.year
train_data['Month'] = train_data['date'].dt.month
train_data['Day']   = train_data['date'].dt.day

test_data['Year']  = test_data['date'].dt.year
test_data['Month'] = test_data['date'].dt.month
test_data['Day']   = test_data['date'].dt.day

# 5) Drop the 'date' column if you no longer need it directly
train_data.drop('date', axis=1, inplace=True)
test_data.drop('date',  axis=1, inplace=True)

# 6) Log-transform the target
train_data['num_sold'] = np.log1p(train_data['num_sold'])

# 7) Drop 'id' from training, we'll use it from test
train_data.drop('id', axis=1, inplace=True)

# Identify numerical and categorical columns
num_cols = train_data.select_dtypes(include=np.number).drop(columns=['num_sold']).columns.tolist()
cat_cols = train_data.select_dtypes(include='object').columns.tolist()

# The test set also has 'id', we'll keep it for submission
# but we won't use it as a feature
test_ids = test_data['id'].copy()  # save for submission
test_data.drop('id', axis=1, inplace=True)

# ----------------- ENCODE CATEGORICAL -------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

    # Transform test data with the same encoder
    if col in test_data.columns:
        test_data[col] = le.transform(test_data[col])

# ----------------- SEPARATE FEATURES/TARGET -------------------
X = train_data.drop(['num_sold'], axis=1)
y = train_data['num_sold']
X_test_final = test_data.copy()  # for final predictions

# ----------------- DEFINE MAPE -------------------
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# ----------------- CROSS-VALIDATION FUNCTIONS -------------------
def cross_val_xgb(X, y, X_test, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    test_preds_list = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred_valid = model.predict(X_valid)
        score = mape(y_valid, y_pred_valid)
        mape_scores.append(score)

        # Predict on the test set
        y_test_pred = model.predict(X_test)
        test_preds_list.append(y_test_pred)

    # Average test predictions across folds
    test_preds_mean = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), test_preds_mean

def cross_val_lgbm(X, y, X_test, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    test_preds_list = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        y_pred_valid = model.predict(X_valid)
        score = mape(y_valid, y_pred_valid)
        mape_scores.append(score)

        # Predict on test
        y_test_pred = model.predict(X_test)
        test_preds_list.append(y_test_pred)

    test_preds_mean = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), test_preds_mean

def cross_val_catboost(X, y, X_test, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    test_preds_list = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

        y_pred_valid = model.predict(X_valid)
        score = mape(y_valid, y_pred_valid)
        mape_scores.append(score)

        # Predict on test
        y_test_pred = model.predict(X_test)
        test_preds_list.append(y_test_pred)

    test_preds_mean = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), test_preds_mean

# ----------------- TRAIN & PREDICT WITH EACH MODEL -------------------
print("=== Training XGB ===")
xgb_cv_score, xgb_test_preds = cross_val_xgb(X, y, X_test_final, xgb_params, n_splits=N_SPLITS)
print(f"XGB - Mean CV MAPE: {xgb_cv_score:.4f}")

print("\n=== Training LGBM ===")
lgb_cv_score, lgb_test_preds = cross_val_lgbm(X, y, X_test_final, lgb_params, n_splits=N_SPLITS)
print(f"LGBM - Mean CV MAPE: {lgb_cv_score:.4f}")

print("\n=== Training CatBoost ===")
cat_cv_score, cat_test_preds = cross_val_catboost(X, y, X_test_final, cat_params, n_splits=N_SPLITS)
print(f"CatBoost - Mean CV MAPE: {cat_cv_score:.4f}")

# ----------------- ENSEMBLE / BLEND -------------------
# A simple approach is an unweighted average
ensemble_test_preds = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3.0

# You could also do a weighted average if one model is much stronger:
# ensemble_test_preds = (0.4 * lgb_test_preds) + (0.3 * xgb_test_preds) + (0.3 * cat_test_preds)

# ----------------- CREATE SUBMISSION -------------------
# Recall we log-transformed the target, so exponentiate predictions
final_num_sold = np.expm1(ensemble_test_preds)

submission = pd.DataFrame({
    'id': test_ids,
    'num_sold': final_num_sold
})

print("\nSample Submission Preview:")
print(submission.head())

submission.to_csv("submission_ensemble.csv", index=False)
print("\nSubmission file 'submission_ensemble.csv' created!")
