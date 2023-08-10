import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task(retries=2, retry_delay_seconds=5, name="read and clean the dataframe")
def read_and_clean_dataframe(filename: str):
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

@task(retries=2, retry_delay_seconds=5, name="remove outliers")
def remove_outliers(df: pd.DataFrame, features: []):
    outlier_indices = []
    for f in features:
        p01 = np.percentile(df[f], 1)
        p99 = np.percentile(df[f], 99)
        outlier_list_col = df[(df[f] < p01) | (df[f] > p99)].index
        outlier_indices.extend(outlier_list_col)
        df_with_outliers_index = df.loc[outlier_indices].index
        df_without_outliers = df.drop(df_with_outliers_index, axis=0)

    return df_without_outliers

@task(retries=2, retry_delay_seconds=5, name="preparing features for the model")
def preprocess(df: pd.DataFrame, dv: DictVectorizer):
    X = df.drop("Price", axis=1)

    categorical = list(X.select_dtypes(include=["int"]).columns)
    numerical = list(X.select_dtypes(exclude=["int"]).columns)

    dicts = X[categorical + numerical].to_dict(orient="records")

    dicts = dv.transform(dicts)

    return dicts


best_params = {
    "n_estimators": int(47),
    "max_depth": int(20),
    "min_samples_split": int(5),
    "min_samples_leaf": int(2),
    "random_state": 42,
    "n_jobs": -1,
}

data_path = "./preprocessed_data"

X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))

model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

@flow
def batch_car_price_prediction():
    raw_data = read_and_clean_dataframe("car_price_prediction.csv")
    features = ["Price", "Mileage"]
    data_without_outliers = remove_outliers(raw_data, features)
    dv = load_pickle(os.path.join(data_path, "dv.pkl"))
    dicts = preprocess(data_without_outliers, dv)

    preds = model.predict(dicts)

    return print(preds)


if __name__ == "__main__":
    batch_car_price_prediction()
