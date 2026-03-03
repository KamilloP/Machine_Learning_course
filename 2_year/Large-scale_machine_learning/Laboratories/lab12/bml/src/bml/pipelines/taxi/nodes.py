import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


NUMERICAL_FEATURES = ["trip_miles", "fare", "trip_seconds"]

BUCKET_FEATURES = [
    "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
]

CATEGORICAL_NUMERICAL_FEATURES = [
    "trip_start_hour", "trip_start_day", "trip_start_month",
    "pickup_census_tract", "dropoff_census_tract",
    "pickup_community_area", "dropoff_community_area",
]

CATEGORICAL_STRING_FEATURES = ["payment_type", "company"]


def prepare_data(df: pd.DataFrame, parameters: dict):
    df = df.copy()
    print(df.describe())
    print(df.dtypes)
    df[NUMERICAL_FEATURES + BUCKET_FEATURES + CATEGORICAL_NUMERICAL_FEATURES + ["tips"]] = df[NUMERICAL_FEATURES + BUCKET_FEATURES + CATEGORICAL_NUMERICAL_FEATURES + ["tips"]].fillna(0)
    df[CATEGORICAL_STRING_FEATURES] = df[CATEGORICAL_STRING_FEATURES].fillna("")

    df["target"] = (df["tips"] > 0.2 * df["fare"]).astype(int)
    df = df.drop(columns=["tips"])

    X = df[NUMERICAL_FEATURES + BUCKET_FEATURES +
           CATEGORICAL_NUMERICAL_FEATURES + CATEGORICAL_STRING_FEATURES]
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"]
    )


def train_model(X_train, X_test, y_train, y_test, parameters):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bucket", KBinsDiscretizer(
                n_bins=parameters["bucket_count"], encode="onehot"), BUCKET_FEATURES),
            ("cat_num", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_NUMERICAL_FEATURES),
            ("cat_str", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_STRING_FEATURES),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    print("Taxi model accuracy:", acc)
    df = pd.DataFrame([[acc]])

    return model, df


import logging
def log_accuracy(accuracy):
    print(accuracy)
    acc = accuracy.values[0,0]
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", acc * 100)
