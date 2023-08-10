import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_and_clean_dataframe(filename: str):
    df = pd.read_csv(filename)
    df = df.drop(["Levy", "ID"], axis="columns")

    # Some data cleaning here
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
    y = df["Price"]
    X = df.drop("Price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )

    categorical = list(X.select_dtypes(include=["int"]).columns)
    numerical = list(X.select_dtypes(exclude=["int"]).columns)

    train_dicts = X_train[categorical + numerical].to_dict(orient="records")
    val_dicts = X_val[categorical + numerical].to_dict(orient="records")
    test_dicts = X_test[categorical + numerical].to_dict(orient="records")

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    X_test = dv.transform(test_dicts)

    return X_train, X_val, X_test, y_train, y_val, y_test


# MODIFY HERE IF NEEDED.
raw_data_path = (
    "C:/Users/sebas/Desktop/mlops-zoomcamp-main/mlops-zoomcamp-sebasfac/Project"
)

dest_path = raw_data_path + "/preprocessed_data"


def run_data_prep(raw_data_path: str, dest_path: str):
    df = read_and_clean_dataframe(
        os.path.join(raw_data_path, "car_price_prediction.csv")
    )
    features = ["Price", "Mileage"]
    df = remove_outliers(df, features)
    dv = DictVectorizer()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, dv)

    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    print("Script was run until the end!")


if __name__ == "__main__":
    run_data_prep(raw_data_path, dest_path)
