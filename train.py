import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

import re
import xgboost as xgb

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


def generate_features(df, current_year=2024) -> pd.DataFrame:
    """
    Generate new features for a house pricing dataset.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with house data.
    - current_year (int): The current year for calculating house age.

    Returns:
    - pd.DataFrame: DataFrame with new features added.
    """
    
    df['house_age'] = current_year - df['year_built']
    df['bed_bath_ratio'] = df['num_bedrooms'] / df['num_bathrooms']
    df['lot_size_per_sqft'] = df['lot_size'] / df['square_footage']
    df['garage_space_per_bedroom'] = df['garage_size'] / df['num_bedrooms']
    df['recently_renovated'] = (df['house_age'] <= 10).astype(int)
    df['modernness_index'] = (10 - df['house_age'] / 10) + df['garage_size'] + df['neighborhood_quality']
    df['outdoor_space'] = df['lot_size'] - (df['square_footage'] / 43560)  # 43560 sq. ft. in an acre
    df['expansion_potential'] = df['lot_size'] * df['neighborhood_quality']
    
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess house pricing data.

    Parameters:
    - file_path (str): The location of the input CSV file.

    Returns:
    - pd.DataFrame: DataFrame with new features added and outliers filtered.
    """
    
    # Load data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    
    # Convert 'house_price' to log scale
    df['house_price'] = np.log1p(df['house_price'])
    
    # Calculate IQR, lower and upper bounds
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    LowerBound = q1 - 1.5 * IQR
    UpperBound = q3 + 1.5 * IQR
    
    # Filter outliers by applying conditions across all columns
    df = df[~((df < LowerBound) | (df > UpperBound)).any(axis=1)]
    
    # Generate new features
    df = generate_features(df)
    
    return df

def split_data(df: pd.DataFrame):
    # Split data into train and test sets
    X = df.drop('house_price', axis=1)
    y = df['house_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_xgb_model(df: pd.DataFrame): 
    """
    Train an XGBoost regressor model on the given DataFrame with the best parameters 
    (chosen by hyperparameter tuning) and save the model to a pickle file.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the house pricing data.

    Returns:
    - None
    """
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    best_params = {'subsample': 0.8, 
                   'n_estimators': 200, 
                   'min_child_weight': 10, 
                   'max_depth': 3, 
                   'learning_rate': 0.1, 
                   'gamma': 0, 
                   'colsample_bytree': 1.0}
    
    # Create the XGBoost regressor with the best parameters
    xgb_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    
    # Create the pipeline
    pipeline = Pipeline([
        ('vectorizer', DictVectorizer()),  # Convert DataFrame to a format suitable for XGBoost
        ('xgb_model', xgb_model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train.to_dict(orient='records'), y_train)
    
    # Predict on the train set
    y_pred = pipeline.predict(X_train.to_dict(orient='records'))
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print("Training RMSE:", rmse)
    
    # Predict on the test set
    y_pred = pipeline.predict(X_test.to_dict(orient='records'))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Testing RMSE:", rmse)
    
    # Save the pipeline
    model_path = 'xgb_pipeline.pkl'
    joblib.dump(pipeline, model_path)
    
    
if __name__ == "__main__":
    df = load_data('dataset/house_price_regression_dataset.csv')
    train_xgb_model(df)