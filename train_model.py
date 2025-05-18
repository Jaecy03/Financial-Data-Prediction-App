"""
Script to train and save the Random Forest model for churn prediction.
This creates a model file that can be loaded by the web app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_and_save_model():
    """Train the Random Forest model and save it to disk."""
    print("Loading data...")
    # Load the dataset
    df = pd.read_csv('data_for_predictions.csv')
    
    # Remove the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
    print("Preparing data...")
    # Separate target variable from independent variables
    y = df['churn']
    X = df.drop(columns=['id', 'churn'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("Training model...")
    # Train the Random Forest model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    print("Saving model...")
    # Create a models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the feature names (column names) for reference
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("Model saved successfully!")
    
    # Return the model and feature names for reference
    return model, X.columns.tolist()

if __name__ == "__main__":
    train_and_save_model()
