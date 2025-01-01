import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from datetime import datetime
import holidays


def calcuate_mape(y_true, y_pred):
    """Calcuate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid zero division
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_features(df, is_train=True):
    """Create time series features from datetime index"""
    df = df.copy()

    # Convert date to datetime if it isn't already
    df["date"] = pd.to_datetime(df["date"])

    # Basic time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int")

    # Create holiday features of each country
    for country in df["country"].unique():
        country_holidays = holidays.CountryHoliday(country)
        df.loc[df["country"] == country, "is_holiday"] = (
            df[df["country"] == country]["date"]
            .map(lambda x: x in country_holidays)
            .astype(int)
        )
    if is_train:
        # Lag features (previous 7, 14, 30 days)
        for lag in [7, 14, 30]:
            df[f"sales_lag_{lag}"] = df.groupby(["country", "store", "product"])[
                "num_sold"
            ].shift(lag)

        # Rolling window features
        for window in [7, 14, 30]:
            df[f"sales_rolling_mean_{window}"] = df.groupby(
                ["country", "store", "product"]
            )["num_sold"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

    return df


def train_model(train_df):
    """Train LightGBM model with time series features"""
    # Create features
    df = create_features(train_df)

    # Encode categorical variables
    le_dict = {}
    for col in ["country", "store", "product"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        le_dict[col] = le

    # Define Features
    feature_cols = [
        "year",
        "month",
        "day",
        "dayofweek",
        "quarter",
        "is_weekend",
        "is_holiday",
        "country_encoded",
        "store_encoded",
        "product_encoded",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_30",
        "sales_rolling_mean_7",
        "sales_rolling_mean_14",
        "sales_rolling_mean_30",
    ]

    # remove rows with NaN values (first month will have NaN for lag features)
    df = df.dropna()

    # Time-based validation split
    train_size = int(len(df) * 0.8)  # using 80% for training
    train_data = df.iloc[:train_size]
    valid_data = df.iloc[train_size:]

    # Split features and target
    X_train = train_data[feature_cols]
    y_train = train_data["num_sold"]
    X_valid = valid_data[feature_cols]
    y_valid = valid_data["num_sold"]

    # Train model
    params = {
        "objective": "regression",
        "metric": "mape",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "colsample_bytree": 0.9,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="mape",
    )

    # Calculate and print validation score
    val_predictions = model.predict(X_valid)
    valid_mape = calcuate_mape(y_valid, val_predictions)
    print(f"\n Validation MAPE: {valid_mape:.2f}%")

    return model, le_dict, feature_cols


def prepare_test_data(test_df, train_df, le_dict):
    """Prepare test data with same features as training data"""
    # Create features
    test_df["num_sold"] = 0
    df = create_features(test_df)

    # Fill missing lag features with training data
    full_data = pd.concat([train_df, test_df], axis=0, sort=False)
    full_data = create_features(full_data, is_train=True)

    # Get the last row only (test_set)
    df = full_data.iloc[-len(test_df) :]

    # Encode categorical variables using same encoder as training
    for col in ["country", "store", "product"]:
        df[f"{col}_encoded"] = le_dict[col].transform(df[col])

    return df


def make_prediction(model, test_df, train_df, le_dict, feature_cols):
    """Make prediction on test data"""
    # Prepare test data
    df = prepare_test_data(test_df, train_df, le_dict)

    # Make preidction
    predictions = model.predict(df[feature_cols])

    # Create submission dataframe
    submission = pd.DataFrame({"id": test_df["id"], "num_sold": predictions})

    return submission


# Main execution
def main():
    # load the data
    path = Path("/home/sonujha/rnd/Forecasting-Sticker-Sales/data/")
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    test_df = pd.read_csv(os.path.join(path, "test.csv"))

    # Train model
    model, le_dict, feature_cols = train_model(train_df)

    # Make Predictions
    submission = make_prediction(model, test_df, train_df, le_dict, feature_cols)

    # Save prediction
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
