import pandas as pd

import car_price_predictor


def test_read_and_clean_dataframe():
    raw_data = car_price_predictor.read_and_clean_dataframe("car_price_prediction.csv")

    actual_ncols = len(raw_data.columns)
    expected_ncols = 17

    assert expected_ncols == actual_ncols


def test_remove_outliers():
    features = ["Price", "Mileage"]
    raw_data = car_price_predictor.read_and_clean_dataframe("car_price_prediction.csv")
    data_without_outliers = car_price_predictor.remove_outliers(raw_data, features)

    initial_df_nrows = len(raw_data)
    expected_nrows = len(data_without_outliers)

    assert expected_nrows < initial_df_nrows
