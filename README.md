# MLOps Zoomcamp Project: Car Price Prediction
Short description: this is the capstone project of the MLOps zoomcamp 2023 edition.

![several-cars](https://github.com/Sebasfac/mlops_zoomcamp_project_car_price_prediction/assets/48665389/a13cd8e2-12f5-42e7-984e-c270feacee2b)


## Overview
The objective of this project is to have a very simple machine learning operations (MLOps) workflow, where we will predict car prices based on their attributes such as the manufacturer, car model, year of production, engine type and other characteristics. Needless to say, price prediction is a very important task in market economies as it allows economic agents to formulate better plans, therefore increasing resource efficiency. The model runs locally so it is not deployed on the cloud. It can take up to 6 GB of disk space.

Jupyter notebooks will be used for initial data exploration while MLflow will be used for model experimentation and registration.

We will also have a basic monitoring system with Evidently AI, Prefect, Grafana and Postgres DB to display a few relevant metrics based on a CSV file and the model predictions.
In the end we will have a python script receiving the mentioned CSV file and outputting reasonable predictions about car prices while being orchestrated in a very basic way with Prefect.

## Dataset
The dataset (CSV) has 19237 rows x 18 columns and comes from the following link:

https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

Sample of the data:

![data sample](https://github.com/Sebasfac/mlops_zoomcamp_project_car_price_prediction/assets/48665389/d70669ac-64d4-41a7-acd2-ecadd2ec878c)


## Tech stack
* Jupyter Notebook
* VS Code
* Python
* MLflow
* Prefect
* Evidently AI
* Grafana
* Docker Compose
* Anaconda Prompt (CLI)

## Instructions
All the dependencies are in the requirements text file.

### Data exploration

The Jupyter Notebooks are just for data exploration and there are no relevant instructions to be given here.

### Data preprocessing

For the creation of the preprocessing python file I used the following command in CLI:

* jupyter nbconvert --to script price_prediction_exploration.ipynb

The file price_prediction_preprocessing.py then produces some pickle files inside the preprocessed_data folder which will be used by the model later.

### Experimentation

Then the following files are to be used with MLflow:

* RF_optimization.py
* RFbest_model.py

The first file is about model experimentation and the second one registers the model in MLflow. Prior to running these files I needed to use the following command in CLI to start MLflow service:

* mlflow ui --backend-store-uri sqlite:///mlflow.db

To access MLflow User Interface (UI) browse to http://127.0.0.1:5000

The folder mlartifacts contains the Mlflow artifacts.

### Monitoring
To calculate the monitoring metrics with evidently_metrics.py, it was created a docker-compose.yaml file for postgres, adminer and grafana services. A prefect server was also used. So, before running the python file I needed to start Docker Desktop and run the following commands in CLI:

* docker-compose up --build
* prefect server start

To access Grafana UI browse to localhost:3000

Grafana default login credentials are admin and admin

The folder config is the configuration of Grafana.

The folder dashboards contains one table with calculated metrics on different subsets of the dataset.

### Predictions
For predicting car prices there is the file car_price_predictor.py which is used with the Prefect server running.
The folder tests contain the respective unit tests which can be run by the CLI with the following command:

* pytest tests/

For the pylint exceptions a file .pylintrc is used.

The Makefile contains a run of a few quality checks too.
