# House Price Prediction

DatatalksClub's ML Zoomcamp Midterm Project -2024

## Problem Description

Predicting house prices is a crucial task in real estate, assisting buyers, sellers, and investors in making informed decisions. This project focuses on creating a machine learning model to estimate house prices based on key features such as square footage, number of bedrooms, lot size, and neighborhood quality.

The solution includes a feature engineering step to enrich the dataset with additional meaningful features and deploys a predictive model via a Flask API for easy access.

## About the Dataset

This beginner-friendly regression dataset, titled `Home Value Insights`, contains 1,000 rows. Each row represents a house with its key attributes that influence its price. The dataset serves as a great starting point for learning and implementing regression models for real-world use cases like house price prediction.

You can find dataset in [`here`](https://www.kaggle.com/datasets/prokshitha/home-value-insights/data).

### Features

1. **Square_Footage**: The size of the house in square feet. Larger homes typically have higher prices.
2. **Num_Bedrooms**: The number of bedrooms in the house. More bedrooms generally increase the value of a home.
3. **Num_Bathrooms**: The number of bathrooms in the house. Houses with more bathrooms are typically priced higher.
4. **Year_Built**: The year the house was built. Older houses may be priced lower due to wear and tear.
5. **Lot_Size**: The size of the lot the house is built on, measured in acres. Larger lots tend to add value to a property.
6. **Garage_Size**: The number of cars that can fit in the garage. Houses with larger garages are usually more expensive.
7. **Neighborhood_Quality**: A rating of the neighborhood’s quality on a scale of 1-10, where 10 indicates a high-quality neighborhood. Better neighborhoods usually command higher prices.
8. **House_Price**: The target variable, representing the price of the house in USD.

## Feature Engineering

To enhance predictive performance, new features were generated based on the original dataset:

1. **house_age**: Age of the house (current_year - year_built).
2. **bed_bath_ratio**: Ratio of the number of bedrooms to bathrooms (num_bedrooms / num_bathrooms).
3. **lot_size_per_sqft**: Lot size relative to the house size (lot_size / square_footage).
4. **garage_space_per_bedroom**: Garage space available per bedroom (garage_size / num_bedrooms).
5. **recently_renovated**: A flag indicating if the house was built or renovated in the last 10 years (house_age <= 10).
6. **modernness_index**: Composite score reflecting modern features ((10 - house_age / 10) + garage_size + neighborhood_quality).
7. **outdoor_space**: The unbuilt portion of the lot (lot_size - (square_footage / 43560) where 43,560 sq. ft. equals 1 acre).
8. **expansion_potential**: A score indicating the house's potential for future expansion (lot_size * neighborhood_quality).


## Project Components

### 1. Model Training and Evaluation

* Trained multiple regression models including:

    * **Linear Regression**
    * **Decision Tree Regressor**
    * **Random Forest Regressor**
    * **XGBoost Regressor**

* Fine-Tuning:
    - **Random Forest Regressor**: Hyperparameters optimized using **GridSearchCV**, exhaustively testing combinations for the best model performance.
    - **XGBoost Regressor**: Optimized with **RandomizedSearchCV**, sampling parameter combinations randomly for faster tuning.

For detailed steps on model training and evaluation, check out [notebook.ipynb](notebook.ipynb).


### 2. Best Model Selection

After evaluating multiple regression models, the **XGBoost Regressor** emerged as the best performer with the lowest RMSE of **0.0401**, indicating the highest predictive accuracy for the house price prediction task.

#### Model Performance (RMSE):

- **Linear Regression**: RMSE = 0.0882
- **Decision Tree Regressor**: RMSE = 0.1115
- **Random Forest Regressor**: RMSE = 0.0571
- **XGBoost Regressor**: RMSE = 0.0401 (Best Model)

### 3. Final Model Training

The **XGBoost Regressor** was selected as the final model due to its superior performance. The model was trained with the best hyperparameters found through **RandomizedSearchCV**. 

```dict
{   
    'subsample': 0.8, 
    'n_estimators': 200, 
    'min_child_weight': 10, 
    'max_depth': 3, 
    'learning_rate': 0.1, 
    'gamma': 0, 
    'colsample_bytree': 1.0
}
```

The full training process is implemented in the [`train.py`](train.py) script, where the final model is trained and saved for deployment in the Flask web application.

### 4. Flask API

* A Flask app serves the prediction API.
* Users send house attributes as input, and the API responds with predicted house prices.

### 5. Docker Containerization

* Packaged the application into a Docker container for easy deployment.


## Installation

1. Clone the Repository

    ```bash
    git clone https://github.com/zwe-htet-paing/house-price-prediction.git
    cd house-price-prediction 
    ```

2. Install Dependencies

* Install `pipenv`:

    ```bash
    pip install pipenv
    ```

* Install required depencies:

    ```bash
    pipenv install
    ```

### Run Locally

1. Start the Flask API:

    ```bash
    pipenv run python predict.py
    ```

2. Open new terminal and run the test:

    ```bash
    pipenv run python test.py
    ```

The API will run locally on `http://0.0.0.0:9696`


## Run on Docker

1. Build the Docker images:

    ```bash
    docker build -t house-price-prediction .
    ```

2. Run the Docker container:

    ```bash
    docker run -it --rm -p 9696:9696 house-price-prediction
    ```

3. Open the new terminal and run the test script:

    ```bash
    pipenv run python test.py
    ```

The API will accessable at `http://localhost:9696/predict`.

## API Usage

### Endpoint: `/predict` (POST)
* Description: Predicts house prices based on input features.
* Input: JSON object with house-related features.
* Output: Predicted house price.

### Sample Request (cURL)

```json
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{
    "square_footage": 3000,
    "num_bedrooms": 5,
    "num_bathrooms": 4,
    "year_built": 2010,
    "lot_size": 0.7,
    "garage_size": 3,
    "neighborhood_quality": 9
}'

```

### Sample Response

```json
{
    "predicted_value":675612.899
}
```

## Future Improvements

- Enhance feature engineering with additional variables like location or market trends.
- Fine-tune the hyperparameters for XGBoost for improved performance.
- Add support for batch predictions.
- Deploy the API to a cloud platform like AWS or GCP.

## License

This project is open-source and free to use under the **GNU General Public License (GPL)** v3.0.

