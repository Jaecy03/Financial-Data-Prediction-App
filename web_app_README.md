# Churn Prediction Web App

This directory contains a simple web interface for making churn predictions using the trained Random Forest model.

## Files

- `train_model.py`: Script to train and save the Random Forest model
- `app.py`: Streamlit web application for making predictions
- `requirements.txt`: List of required Python packages

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train and save the model:
   ```
   python train_model.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Using the Web App

1. Enter customer information in the form fields
2. Click "Predict Churn" to get a prediction
3. View the prediction result and recommendations
4. Explore the "Model Information" tab to understand feature importance

## Features

- User-friendly interface for entering customer data
- Real-time prediction of churn probability
- Visual representation of prediction results
- Recommendations based on prediction outcome
- Model information and feature importance visualization

## Technical Details

The web app uses:
- Streamlit for the user interface
- Scikit-learn for the machine learning model
- Matplotlib and Seaborn for visualizations
- Pickle for model serialization
