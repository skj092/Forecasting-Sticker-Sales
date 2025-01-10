import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from category_encoders import TargetEncoder
import optuna
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback, CatBoostPruningCallback
import warnings
warnings.filterwarnings('ignore')

# Set a random seed for reproducibility
RANDOM_STATE = 42

# --------------------------------------
# 1. Load Data
# --------------------------------------
path = Path('data')
train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'test.csv')
sub = pd.read_csv(path/'sample_submission.csv')

# Drop rows where target is missing
train = train.dropna(subset=['num_sold'])

# --------------------------------------
# 2. Basic Preprocessing
# --------------------------------------


def transform_date(df, col='date'):
    df[col] = pd.to_datetime(df[col])
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_day_of_week'] = df[col].dt.dayofweek
    df[f'{col}_week_of_year'] = df[col].dt.isocalendar().week

    # Cyclical features
    df[f'{col}_day_sin'] = np.sin(
        2 * np.pi * df[f'{col}_day'] / 31.0)  # day out of max 31
    df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[f'{col}_day'] / 31.0)
    df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12.0)
    df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12.0)

    # Example grouping feature
    df[f'{col}_group'] = (df[f'{col}_year'] - 2010) * 12 + df[f'{col}_month']
    return df


train = transform_date(train, 'date')
test = transform_date(test, 'date')

# Drop original date & ID if not needed
drop_cols = ['date', 'id']
if set(drop_cols).issubset(train.columns):
    train.drop(columns=drop_cols, inplace=True)
if set(drop_cols).issubset(test.columns):
    test.drop(columns=drop_cols, inplace=True)

# --------------------------------------
# 3. Feature/Target Setup
# --------------------------------------
# Log-transform the target
train['num_sold'] = np.log1p(train['num_sold'])

# Split into features & target
X_full = train.drop(columns=['num_sold'])
y_full = train['num_sold']

# We will encode test data similarly
X_test = test.copy()

# Identify categorical columns
cat_cols = X_full.select_dtypes(include=['object']).columns.tolist()

# --------------------------------------
# 4. Advanced Encoding: Target Encoding
# --------------------------------------
encoder = TargetEncoder(cols=cat_cols)
X_full = encoder.fit_transform(X_full, y_full)
X_test = encoder.transform(X_test)

# Optionally, ensure numeric dtypes
for col in X_full.columns:
    if X_full[col].dtype == 'float64':
        X_full[col] = X_full[col].astype('float32')
for col in X_test.columns:
    if X_test[col].dtype == 'float64':
        X_test[col] = X_test[col].astype('float32')

# --------------------------------------
# 5. Optuna Hyperparameter Tuning
#    with possibility to try different models
# --------------------------------------


def objective(trial):
    model_choice = trial.suggest_categorical(
        'model_choice', ['lgbm', 'xgb', 'catboost'])

    # Common hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 500, 2000)
    lr = trial.suggest_loguniform('learning_rate', 0.01, 0.3)

    # Shared cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mape_scores = []

    for train_idx, valid_idx in kf.split(X_full):
        X_train_cv, X_valid_cv = X_full.iloc[train_idx], X_full.iloc[valid_idx]
        y_train_cv, y_valid_cv = y_full.iloc[train_idx], y_full.iloc[valid_idx]

        if model_choice == 'lgbm':
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mape',
                'n_estimators': n_estimators,
                'learning_rate': lr,
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'random_state': RANDOM_STATE
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_valid_cv, y_valid_cv)],
                eval_metric='mape',
            )

        elif model_choice == 'xgb':
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mape',
                'n_estimators': n_estimators,
                'learning_rate': lr,
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1.0),
                'random_state': RANDOM_STATE,
                'verbosity': 0
            }
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_valid_cv, y_valid_cv)],
            )

        else:  # CatBoost
            params = {
                'iterations': n_estimators,
                'learning_rate': lr,
                'depth': trial.suggest_int('depth', 5, 15),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 1.0),
                'random_seed': RANDOM_STATE,
                'logging_level': 'Silent',
                'loss_function': 'MAPE'
            }
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train_cv, y_train_cv,
                eval_set=(X_valid_cv, y_valid_cv),
            )

        y_pred_cv = model.predict(X_valid_cv)
        fold_mape = mean_absolute_percentage_error(y_valid_cv, y_pred_cv)
        mape_scores.append(fold_mape)

    return np.mean(mape_scores)


# Create study with pruning
study = optuna.create_study(
    direction='minimize', study_name='Sales Forecasting')
# Increase n_trials if possible
study.optimize(objective, n_trials=100, timeout=3600)

print("Best params:", study.best_params)
print("Best MAPE:", study.best_value)

# --------------------------------------
# 6. Train Final Model (Ensemble or Single Best)
# --------------------------------------
best_model_name = study.best_params['model_choice']
best_params = {k: v for k, v in study.best_params.items() if k not in [
    'model_choice']}

# Refit using the entire dataset with cross-validation predictions
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X_full))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_full)):
    print(f"Fold {fold+1}")
    X_train_cv, X_valid_cv = X_full.iloc[train_idx], X_full.iloc[valid_idx]
    y_train_cv, y_valid_cv = y_full.iloc[train_idx], y_full.iloc[valid_idx]

    if best_model_name == 'lgbm':
        final_model = lgb.LGBMRegressor(
            **best_params, random_state=RANDOM_STATE)
    elif best_model_name == 'xgb':
        final_model = xgb.XGBRegressor(
            **best_params, random_state=RANDOM_STATE)
    else:
        # CatBoost
        final_model = cb.CatBoostRegressor(
            **best_params, random_seed=RANDOM_STATE, verbose=False)

    final_model.fit(X_train_cv, y_train_cv)
    oof_preds[valid_idx] = final_model.predict(X_valid_cv)

    # Predict on test for each fold; average later
    test_preds += final_model.predict(X_test) / kf.n_splits

fold_mape = mean_absolute_percentage_error(y_full, oof_preds)
print("OOF MAPE:", fold_mape)

# --------------------------------------
# 7. Prepare Submission
# --------------------------------------
# Inverse log transform
sub['num_sold'] = np.expm1(test_preds)
sub.to_csv("submission.csv", index=False)
sub.head()

