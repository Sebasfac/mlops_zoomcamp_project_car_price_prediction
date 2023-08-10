# MLOps Zoomcamp Project: Car Price Prediction
Short description: This is the capstone project of MLOps zoomcamp 2023 edition.

![several-cars](https://github.com/Sebasfac/mlops_zoomcamp_project_car_price_prediction/assets/48665389/a13cd8e2-12f5-42e7-984e-c270feacee2b)


## Overview
The objective of this project is to have a very simple machine learning operations (MLOps) workflow, where we will predict car prices based on their attributes such as the manufacturer, model, year of production, engine type and other characteristics. The model runs locally and is not deployed on the cloud. It can take around 6 gb of disk space.
Jupyter notebooks will be used for initial data exploration while later MLflow will be used for model experimentation and registration.

In the end we will have a python script receiving a csv file and throwing out reasonable predictions about car prices while being orchestrated in a very basic way with Prefect. We will also have a basic monitoring system with Evidently AI, Prefect, Grafana and Postgres DB to throw some quality metrics based on the mentioned csv file and the model predictions.

## Dataset
The dataset has 19237 rows x 18 columns and is the following:

https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

Sample of the data:

![data sample](https://github.com/Sebasfac/mlops_zoomcamp_project_car_price_prediction/assets/48665389/d70669ac-64d4-41a7-acd2-ecadd2ec878c)


## Tech stack
* Jupyter Notebook
* Python
* MLflow
* Prefect
* Evidently AI
* Grafana
* Docker
* Anaconda Prompt (CLI)

## Instructions
All the dependencies are in the requirements file.


The Jupyter Notebooks are just for data exploration and there are no relevant instructions to be given here.


For the creation of the preprocessing python file I used the following command in CLI:

jupyter nbconvert --to script price_prediction_exploration.ipynb


Then the following files are to be used with MLflow:

* RF_optimization.py
* RFbest_model.py



The first file is about model experimentation and the second one registers the model in MLflow. Prior to running these files I needed to use the following command in CLI to start MLflow service:

mlflow ui --backend-store-uri sqlite:///mlflow.db

To calculate the monitoring metrics with evidently_metrics.py, it was created a docker-compose.yaml file for postgres, adminer and grafana services. A prefect server was also used. So, before running the python file I needed to start Docker Desktop and run the following commands in CLI:

docker-compose up --build
prefect server start

To access Grafana UI browse to localhost:3000
Grafana default login credentials are admin and admin

The folder dashboards contain one table with calculated metrics on different subsets of the dataset.
