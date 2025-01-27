import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import optuna
import warnings
warnings.filterwarnings('ignore')

path = Path('data')
train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'test.csv')
sub = pd.read_csv(path/'sample_submission.csv')

train = train.dropna()
test = test.dropna()


def transform_date(df, col):
    # Convert the column to datetime
    df[col] = pd.to_datetime(df[col])

    # Extract temporal features
    df[f'{col}_year'] = df[col].dt.year.astype('float64')
    df[f'{col}_quarter'] = df[col].dt.quarter.astype('float64')
    df[f'{col}_month'] = df[col].dt.month.astype('float64')
    df[f'{col}_day'] = df[col].dt.day.astype('float64')
    df[f'{col}_day_of_week'] = df[col].dt.dayofweek.astype('float64')
    df[f'{col}_week_of_year'] = df[col].dt.isocalendar().week.astype('float64')
    df[f'{col}_hour'] = df[col].dt.hour.astype('float64')
    df[f'{col}_minute'] = df[col].dt.minute.astype('float64')

    # Add cyclical encodings
    df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[f'{col}_day'] / 365.0)
    df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[f'{col}_day'] / 365.0)
    df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12.0)
    df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12.0)
    df[f'{col}_year_sin'] = np.sin(2 * np.pi * df[f'{col}_year'] / 7.0)
    df[f'{col}_year_cos'] = np.cos(2 * np.pi * df[f'{col}_year'] / 7.0)

    # Add group feature (for time-based grouping)
    df[f'{col}_Group'] = (df[f'{col}_year'] - 2010) * \
        48 + df[f'{col}_month'] * 4 + df[f'{col}_day'] // 7

    return df


new_train = transform_date(train, 'date')
new_test = transform_date(test, 'date')

new_train['num_sold'] = np.log1p(new_train['num_sold'])
new_train = new_train.drop(columns=['date', 'id'], axis=1)
new_test = new_test.drop(columns=['date', 'id'], axis=1)


num_cols = list(new_train.select_dtypes(
    exclude=['object']).columns.difference(['num_sold']))
cat_ftrs = list(new_train.select_dtypes(include=['object']).columns)

num_cols_test = list(new_test.select_dtypes(
    exclude=['object']).columns.difference(['id']))
cat_ftrs_test = list(new_test.select_dtypes(include=['object']).columns)

train_test_comb = pd.concat([new_train, new_test], axis=0, ignore_index=True)
for col in cat_ftrs:
    train_test_comb[col], _ = train_test_comb[col].factorize()
    train_test_comb[col] -= train_test_comb[col].min()
    # label encode to categorical and convert int32 to category
    train_test_comb[col] = train_test_comb[col].astype('int32')
    train_test_comb[col] = train_test_comb[col].astype('category')

for col in num_cols:
    if train_test_comb[col].dtype == 'float64':
        train_test_comb[col].astype('float32')
    if train_test_comb[col].dtype == 'int64':
        train_test_comb[col].astype('int32')

new_train = train_test_comb.iloc[:len(new_train)].copy()
new_test = train_test_comb.iloc[len(new_train):].copy()

new_test = new_test.drop(columns='num_sold', axis=1)

# Calculate the correlation matrix
correlation_matrix = new_train.corr()

X = new_train.drop(columns=['num_sold'])
y = new_train['num_sold']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mape',  # We'll evaluate on MAPE
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'random_state': 42,
        'verbose': -1,
        'device': 'cpu'
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    return mape


# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best parameters and MAPE
print("Best parameters:", study.best_params)
print("Best MAPE:", study.best_value)

# Use the best parameters found by Optuna for final training and prediction
lgb_params = study.best_params
lgb_params.update({
    'device': 'cpu',                # Use GPU for training
    'n_jobs': -1,                   # Use all available CPU threads
})

# K-Fold Cross-validation with LightGBM
scores, lgb_test_preds = [], []

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f'Fold {i}')
    X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Train the model with the best parameters
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)])

    y_preds = lgb_model.predict(X_val_fold)
    mape_score = mean_absolute_percentage_error(y_val_fold, y_preds)
    print(f'MAPE Score for fold {i}:', mape_score)
    scores.append(mape_score)
    lgb_test_preds.append(lgb_model.predict(X))

# Calculate mean and std of MAPE
lgb_score = np.mean(scores)
lgb_std = np.std(scores)

print(f"Mean MAPE: {lgb_score}, Std MAPE: {lgb_std}")

sub['num_sold'] = np.expm1(lgb_model.predict(new_test))
sub.to_csv('submission.csv', index=False)
print(sub.head())
