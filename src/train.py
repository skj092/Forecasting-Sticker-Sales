from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

# Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ================= CONFIG =================
# Hyperparams - adjust or tune for best results
with open('config.yaml') as yf:
    config = yaml.safe_load(yf)
xgb_params, lgb_params, cat_params = config['models']


# =============== READ DATA ===============
path = Path("data")
train_data = pd.read_csv(path / 'train.csv', parse_dates=['date'])
test_data = pd.read_csv(path / 'test.csv', parse_dates=['date'])
sample_sub = pd.read_csv(path / 'sample_submission.csv')

print("Initial train_data shape:", train_data.shape)
print("Initial test_data shape: ", test_data.shape)

# =============== CLEAN & PREPARE TRAIN ===============
# 1) Drop duplicates & missing target
train_data.drop_duplicates(inplace=True)
train_data.dropna(subset=['num_sold'], inplace=True)

# 3) Sort by date for time-series logic
train_data = train_data.sort_values('date').reset_index(drop=True)
test_data = test_data.sort_values('date').reset_index(drop=True)

# =============== FEATURE ENGINEERING ===============


def create_date_features(df):
    # Basic date parts
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Quarter'] = df['date'].dt.quarter
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Cyclical Features
    df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['year_sin'] = np.sin(2 * np.pi * df['Year'] / 7.0)
    df['year_cos'] = np.cos(2 * np.pi * df['Year'] / 7.0)

    # Group Calculation
    df['Group'] = (df['Year'] - 2010) * 48 + df['Month'] * 4 + df['Day'] // 7

    return df


train_data = create_date_features(train_data)
test_data = create_date_features(test_data)

# 4) Lag & Rolling (grouped)
#    We'll do lag_7 and rolling_7 for train only
group_cols = ['country', 'store', 'product']
train_data['lag_7'] = train_data.groupby(group_cols)['num_sold'].shift(7)
train_data['rolling_7'] = (
    train_data.groupby(group_cols)['num_sold'].shift(1).rolling(7).mean()
)
# Drop rows that are NaN due to lag/rolling
train_data.dropna(subset=['lag_7', 'rolling_7'], inplace=True)

# 5) Log transform the target
train_data['num_sold'] = np.log1p(train_data['num_sold'])


# 6) Some columns exist in train but not in test (lag_7, rolling_7),
#    so let's add placeholder columns to the test data so XGB won't complain:
for col in ['lag_7', 'rolling_7']:
    if col not in test_data.columns:
        test_data[col] = 0.0  # simple placeholder

# 7) Drop date column if not needed
train_data.drop('date', axis=1, inplace=True, errors='ignore')
test_data.drop('date', axis=1, inplace=True, errors='ignore')

# 8) Drop 'id' from train; keep test IDs for submission
if 'id' in train_data.columns:
    train_data.drop('id', axis=1, inplace=True)

test_ids = None
if 'id' in test_data.columns:
    test_ids = test_data['id'].copy()
    test_data.drop('id', axis=1, inplace=True)

# Identify numeric & categorical columns
num_cols = train_data.select_dtypes(include=np.number).drop(
    columns=['num_sold']).columns.tolist()
cat_cols = train_data.select_dtypes(include='object').columns.tolist()

# Encode categoricals
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le
    if col in test_data.columns:
        test_data[col] = le.transform(test_data[col])

# Final features & target
X = train_data.drop(['num_sold'], axis=1)
y = train_data['num_sold']
X_test_final = test_data.copy()

print("Final train shape:", X.shape, "y shape:", y.shape)
print("Test shape:", X_test_final.shape)


# =============== CV FUNCTIONS ===============


def cross_val_xgb(X, y, X_test, params):
    mape_scores = []
    test_preds_list = []

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_valid_log = model.predict(X_valid)
    score = mean_absolute_percentage_error(y_valid, y_pred_valid_log)
    mape_scores.append(score)

    y_test_pred_log = model.predict(X_test)
    # Convert from log scale to original scale for final preds
    test_preds_list.append(np.expm1(y_test_pred_log))

    avg_test_preds = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), avg_test_preds


def cross_val_lgbm(X, y, X_test, params):
    mape_scores = []
    test_preds_list = []

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_valid_log = model.predict(X_valid)
    score = mean_absolute_percentage_error(y_valid, y_pred_valid_log)
    mape_scores.append(score)

    y_test_pred_log = model.predict(X_test)
    test_preds_list.append(np.expm1(y_test_pred_log))

    avg_test_preds = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), avg_test_preds


def cross_val_catboost(X, y, X_test, params):
    mape_scores = []
    test_preds_list = []

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

    y_pred_valid_log = model.predict(X_valid)
    score = mean_absolute_percentage_error(y_valid, y_pred_valid_log)
    mape_scores.append(score)

    y_test_pred_log = model.predict(X_test)
    test_preds_list.append(np.expm1(y_test_pred_log))

    avg_test_preds = np.mean(test_preds_list, axis=0)
    return np.mean(mape_scores), avg_test_preds

# =============== TRAIN & PREDICT ===============


print("\n=== Training XGB ===")
xgb_cv_score, xgb_test_preds = cross_val_xgb(
    X, y, X_test_final, xgb_params)
print(f"XGB - Mean CV MAPE (original scale): {xgb_cv_score * 100:.4f}%")

print("\n=== Training LGBM ===")
lgb_cv_score, lgb_test_preds = cross_val_lgbm(
    X, y, X_test_final, lgb_params)
print(f"LGBM - Mean CV MAPE (original scale): {lgb_cv_score * 100:.4f}%")

print("\n=== Training CatBoost ===")
cat_cv_score, cat_test_preds = cross_val_catboost(
    X, y, X_test_final, cat_params)
print(f"CatBoost - Mean CV MAPE (original scale): {cat_cv_score * 100:.4f}%")

# =============== ENSEMBLE ===============
# Simple unweighted average of the three model predictions (already in original scale)
ensemble_test_preds = (xgb_test_preds + lgb_test_preds + cat_test_preds) / 3.0

# If one model is consistently better in local CV, do a weighted average:
ensemble_test_preds = 0.5*lgb_test_preds + 0.3*xgb_test_preds + 0.2*cat_test_preds

# =============== SUBMISSION ===============
if test_ids is None:
    print("No 'id' column found in test data. Cannot create submission with IDs.")
else:
    submission = pd.DataFrame(
        {'id': test_ids, 'num_sold': ensemble_test_preds})
    print("\nSample Submission Preview:")
    print(submission.head())

    submission.to_csv("submission_ensemble.csv", index=False)
    print("\nSubmission file 'submission_ensemble.csv' created!")

