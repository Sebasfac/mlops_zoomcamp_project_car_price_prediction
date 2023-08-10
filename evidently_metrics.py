import datetime
import logging
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import psycopg
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnQuantileMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def read_and_clean_dataframe(filename: str, ref: bool = False):
    half_length = int(len(pd.read_csv("car_price_prediction.csv")) / 2)

    if ref == True:
        df = pd.read_csv(filename)[half_length:]
    else:
        df = pd.read_csv(filename)
    df = df.drop(["Levy", "ID"], axis="columns")

    df["Leather interior"].replace({"Yes": True, "No": False}, inplace=True)

    df["Engine volume"] = df["Engine volume"].str.lower()
    df["Turbo"] = df["Engine volume"].str.contains("turbo")

    df["Engine volume"] = df["Engine volume"].str.slice(0, 3)
    df["Engine volume"] = df["Engine volume"].astype("float64")

    df["Mileage"] = df["Mileage"].str.strip("km")
    df["Mileage"] = df["Mileage"].astype("int64")

    df["Doors"].replace({"04-May": 4, "02-Mar": 2, ">5": 5}, inplace=True)

    return df


def remove_outliers(df: pd.DataFrame, features: list()):
    outlier_indices = []
    for f in features:
        P01 = np.percentile(df[f], 1)
        P99 = np.percentile(df[f], 99)
        outlier_list_col = df[(df[f] < P01) | (df[f] > P99)].index
        outlier_indices.extend(outlier_list_col)
        df_with_outliers_index = df.loc[outlier_indices].index
        df_without_outliers = df.drop(df_with_outliers_index, axis=0)

    return df_without_outliers


def preprocess(df: pd.DataFrame, dv: DictVectorizer):
    X = df.drop("Price", axis=1)

    categorical = list(X.select_dtypes(include=["int"]).columns)
    numerical = list(X.select_dtypes(exclude=["int"]).columns)

    dicts = X[categorical + numerical].to_dict(orient="records")

    dicts = dv.transform(dicts)

    return X, dicts


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists car_metrics;
create table car_metrics(
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	prod_year_median float,
    mileage_median float,
    doors_median float,
    airbags_median float
)
"""


# I should have loaded the model best parameters instead of manually inputting them.
params = {
    "n_estimators": int(47),
    "max_depth": int(20),
    "min_samples_split": int(5),
    "min_samples_leaf": int(2),
    "random_state": 42,
    "n_jobs": -1,
}

data_path = "./preprocessed_data"

X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))


model = RandomForestRegressor(**params)
model.fit(X_train, y_train)


raw_data = read_and_clean_dataframe("car_price_prediction.csv")
features = ["Price", "Mileage"]
raw_data = remove_outliers(raw_data, features)
dv = load_pickle(os.path.join(data_path, "dv.pkl"))
raw_data, dicts = preprocess(raw_data, dv)


reference_data = read_and_clean_dataframe("car_price_prediction.csv", ref=True)
reference_data = remove_outliers(reference_data, features)
reference_data, ref_dicts = preprocess(reference_data, dv)


num_features = ["Prod. year", "Mileage", "Doors", "Airbags"]
cat_features = [
    "Manufacturer",
    "Model",
    "Category",
    "Leather interior",
    "Fuel type",
    "Engine volume",
    "Cylinders",
    "Gear box type",
    "Drive wheels",
    "Wheel",
    "Color",
    "Turbo",
]

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="Prod. year", quantile=0.5),
        ColumnQuantileMetric(column_name="Mileage", quantile=0.5),
        ColumnQuantileMetric(column_name="Doors", quantile=0.5),
        ColumnQuantileMetric(column_name="Airbags", quantile=0.5),
    ]
)


@task(retries=2, retry_delay_seconds=5, name="prepare database")
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)


@task(retries=1, retry_delay_seconds=5, name="calculate metrics")
def calculate_metrics_postgresql(curr, i):
    length = raw_data.shape[0]
    split = int(length / 3)  # Let's split current_data in 3 parts
    starting_point = (i - 1) * split
    ending_point = i * split
    current_data = raw_data[starting_point:ending_point]

    raw_data["prediction"] = model.predict(dicts)
    reference_data["prediction"] = model.predict(ref_dicts)

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    prod_year_median = result["metrics"][3]["result"]["current"]["value"]
    mileage_median = result["metrics"][4]["result"]["current"]["value"]
    doors_median = result["metrics"][5]["result"]["current"]["value"]
    airbags_median = result["metrics"][6]["result"]["current"]["value"]

    curr.execute(
        "insert into car_metrics(prediction_drift, num_drifted_columns, share_missing_values, prod_year_median, mileage_median, doors_median, airbags_median) values (%s, %s, %s, %s, %s, %s, %s)",
        (
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            prod_year_median,
            mileage_median,
            doors_median,
            airbags_median,
        ),
    )


@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for i in range(1, 4):  # 1, 2 and 3
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
