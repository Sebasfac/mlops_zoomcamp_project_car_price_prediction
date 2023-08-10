# mlops_zoomcamp_project_car_price_prediction
This is the capstone project of mlops zoomcamp 2023 edition.

![several-cars](https://github.com/Sebasfac/mlops_zoomcamp_project_car_price_prediction/assets/48665389/a13cd8e2-12f5-42e7-984e-c270feacee2b)


## Overview

## Dataset

Sample of the data:

[
	{
		"metadata": {
			"outputType": "execute_result",
			"executionCount": 77,
			"metadata": {}
		},
		"outputItems": [
			{
				"mimeType": "text/html",
				"data": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Price</th>\n      <th>Manufacturer</th>\n      <th>Model</th>\n      <th>Prod. year</th>\n      <th>Category</th>\n      <th>Leather interior</th>\n      <th>Fuel type</th>\n      <th>Engine volume</th>\n      <th>Mileage</th>\n      <th>Cylinders</th>\n      <th>Gear box type</th>\n      <th>Drive wheels</th>\n      <th>Doors</th>\n      <th>Wheel</th>\n      <th>Color</th>\n      <th>Airbags</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>13328</td>\n      <td>LEXUS</td>\n      <td>RX 450</td>\n      <td>2010</td>\n      <td>Jeep</td>\n      <td>Yes</td>\n      <td>Hybrid</td>\n      <td>3.5</td>\n      <td>186005 km</td>\n      <td>6.0</td>\n      <td>Automatic</td>\n      <td>4x4</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>Silver</td>\n      <td>12</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>16621</td>\n      <td>CHEVROLET</td>\n      <td>Equinox</td>\n      <td>2011</td>\n      <td>Jeep</td>\n      <td>No</td>\n      <td>Petrol</td>\n      <td>3</td>\n      <td>192000 km</td>\n      <td>6.0</td>\n      <td>Tiptronic</td>\n      <td>4x4</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>Black</td>\n      <td>8</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>8467</td>\n      <td>HONDA</td>\n      <td>FIT</td>\n      <td>2006</td>\n      <td>Hatchback</td>\n      <td>No</td>\n      <td>Petrol</td>\n      <td>1.3</td>\n      <td>200000 km</td>\n      <td>4.0</td>\n      <td>Variator</td>\n      <td>Front</td>\n      <td>04-May</td>\n      <td>Right-hand drive</td>\n      <td>Black</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>3607</td>\n      <td>FORD</td>\n      <td>Escape</td>\n      <td>2011</td>\n      <td>Jeep</td>\n      <td>Yes</td>\n      <td>Hybrid</td>\n      <td>2.5</td>\n      <td>168966 km</td>\n      <td>4.0</td>\n      <td>Automatic</td>\n      <td>4x4</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>White</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>11726</td>\n      <td>HONDA</td>\n      <td>FIT</td>\n      <td>2014</td>\n      <td>Hatchback</td>\n      <td>Yes</td>\n      <td>Petrol</td>\n      <td>1.3</td>\n      <td>91901 km</td>\n      <td>4.0</td>\n      <td>Automatic</td>\n      <td>Front</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>Silver</td>\n      <td>4</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>19232</th>\n      <td>8467</td>\n      <td>MERCEDES-BENZ</td>\n      <td>CLK 200</td>\n      <td>1999</td>\n      <td>Coupe</td>\n      <td>Yes</td>\n      <td>CNG</td>\n      <td>2.0 Turbo</td>\n      <td>300000 km</td>\n      <td>4.0</td>\n      <td>Manual</td>\n      <td>Rear</td>\n      <td>02-Mar</td>\n      <td>Left wheel</td>\n      <td>Silver</td>\n      <td>5</td>\n    </tr>\n    <tr>\n      <th>19233</th>\n      <td>15681</td>\n      <td>HYUNDAI</td>\n      <td>Sonata</td>\n      <td>2011</td>\n      <td>Sedan</td>\n      <td>Yes</td>\n      <td>Petrol</td>\n      <td>2.4</td>\n      <td>161600 km</td>\n      <td>4.0</td>\n      <td>Tiptronic</td>\n      <td>Front</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>Red</td>\n      <td>8</td>\n    </tr>\n    <tr>\n      <th>19234</th>\n      <td>26108</td>\n      <td>HYUNDAI</td>\n      <td>Tucson</td>\n      <td>2010</td>\n      <td>Jeep</td>\n      <td>Yes</td>\n      <td>Diesel</td>\n      <td>2</td>\n      <td>116365 km</td>\n      <td>4.0</td>\n      <td>Automatic</td>\n      <td>Front</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>Grey</td>\n      <td>4</td>\n    </tr>\n    <tr>\n      <th>19235</th>\n      <td>5331</td>\n      <td>CHEVROLET</td>\n      <td>Captiva</td>\n      <td>2007</td>\n      <td>Jeep</td>\n      <td>Yes</td>\n      <td>Diesel</td>\n      <td>2</td>\n      <td>51258 km</td>\n      <td>4.0</td>\n      <td>Automatic</td>\n      <td>Front</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>Black</td>\n      <td>4</td>\n    </tr>\n    <tr>\n      <th>19236</th>\n      <td>470</td>\n      <td>HYUNDAI</td>\n      <td>Sonata</td>\n      <td>2012</td>\n      <td>Sedan</td>\n      <td>Yes</td>\n      <td>Hybrid</td>\n      <td>2.4</td>\n      <td>186923 km</td>\n      <td>4.0</td>\n      <td>Automatic</td>\n      <td>Front</td>\n      <td>04-May</td>\n      <td>Left wheel</td>\n      <td>White</td>\n      <td>12</td>\n    </tr>\n  </tbody>\n</table>\n<p>19237 rows × 16 columns</p>\n</div>"
			},
			{
				"mimeType": "text/plain",
				"data": "       Price   Manufacturer    Model  Prod. year   Category Leather interior  \\\n0      13328          LEXUS   RX 450        2010       Jeep              Yes   \n1      16621      CHEVROLET  Equinox        2011       Jeep               No   \n2       8467          HONDA      FIT        2006  Hatchback               No   \n3       3607           FORD   Escape        2011       Jeep              Yes   \n4      11726          HONDA      FIT        2014  Hatchback              Yes   \n...      ...            ...      ...         ...        ...              ...   \n19232   8467  MERCEDES-BENZ  CLK 200        1999      Coupe              Yes   \n19233  15681        HYUNDAI   Sonata        2011      Sedan              Yes   \n19234  26108        HYUNDAI   Tucson        2010       Jeep              Yes   \n19235   5331      CHEVROLET  Captiva        2007       Jeep              Yes   \n19236    470        HYUNDAI   Sonata        2012      Sedan              Yes   \n\n      Fuel type Engine volume    Mileage  Cylinders Gear box type  \\\n0        Hybrid           3.5  186005 km        6.0     Automatic   \n1        Petrol             3  192000 km        6.0     Tiptronic   \n2        Petrol           1.3  200000 km        4.0      Variator   \n3        Hybrid           2.5  168966 km        4.0     Automatic   \n4        Petrol           1.3   91901 km        4.0     Automatic   \n...         ...           ...        ...        ...           ...   \n19232       CNG     2.0 Turbo  300000 km        4.0        Manual   \n19233    Petrol           2.4  161600 km        4.0     Tiptronic   \n19234    Diesel             2  116365 km        4.0     Automatic   \n19235    Diesel             2   51258 km        4.0     Automatic   \n19236    Hybrid           2.4  186923 km        4.0     Automatic   \n\n      Drive wheels   Doors             Wheel   Color  Airbags  \n0              4x4  04-May        Left wheel  Silver       12  \n1              4x4  04-May        Left wheel   Black        8  \n2            Front  04-May  Right-hand drive   Black        2  \n3              4x4  04-May        Left wheel   White        0  \n4            Front  04-May        Left wheel  Silver        4  \n...            ...     ...               ...     ...      ...  \n19232         Rear  02-Mar        Left wheel  Silver        5  \n19233        Front  04-May        Left wheel     Red        8  \n19234        Front  04-May        Left wheel    Grey        4  \n19235        Front  04-May        Left wheel   Black        4  \n19236        Front  04-May        Left wheel   White       12  \n\n[19237 rows x 16 columns]"
			}
		]
	}
]

## Tech stack

## Instructions
