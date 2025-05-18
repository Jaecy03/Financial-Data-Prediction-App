# Financial Data Prediction App

This repository contains a data science project focused on analyzing financial data, performing exploratory data analysis (EDA), engineering features for predictive modeling, and implementing machine learning models for churn prediction.

## Project Structure

The project is organized into four main components:

### Main Directory
- `Task 3 - feature_engineering.ipynb`: Jupyter notebook containing feature engineering processes
- `Task_4_modeling_completed.ipynb`: Jupyter notebook with Random Forest model for churn prediction
- `generate_features.py`: Python script for generating features from the dataset
- `simple_feature_analysis.py`: Python script for basic feature analysis
- `feature_engineering_summary.md`: Summary of the feature engineering process and findings
- `clean_data_after_eda.csv`: Cleaned dataset after exploratory data analysis
- `engineered_features.csv`: Dataset with engineered features
- `feature_correlations.csv`: Correlation matrix of features
- `price_data.csv`: Financial price data
- `data_for_predictions.csv`: Dataset used for churn prediction modeling

### EDA Directory
- `EDA_Analysis.ipynb`: Jupyter notebook containing exploratory data analysis
- `Task 2 - eda_starter.ipynb`: Starter notebook for EDA tasks
- `Data Description.pdf`: Documentation describing the dataset
- `client_data.csv`: Raw client data
- `price_data.csv`: Raw price data

### Modeling Directory
- Contains the churn prediction model implementation using Random Forest

### Web Application
- `app.py`: Streamlit web application for making churn predictions
- `train_model.py`: Script to train and save the Random Forest model
- `app_demo.py`: Demo script showing model functionality
- `streamlit_app_mockup.md`: Visual representation of the web interface
- `requirements.txt`: List of required Python packages
- `models/`: Directory containing the trained model and feature names
- `feature_importance.png`: Visualization of feature importance

## Project Overview

This project involves analyzing financial data to identify patterns and relationships that can be used for predictive modeling. The workflow includes:

1. **Exploratory Data Analysis (EDA)**: Understanding the data structure, identifying patterns, and cleaning the dataset
2. **Feature Engineering**: Creating new features from existing data to improve model performance
3. **Feature Analysis**: Analyzing the importance and relationships between features
4. **Predictive Modeling**: Building a Random Forest classifier to predict customer churn
5. **Web Interface**: Creating a user-friendly application for making predictions

## Churn Prediction Model

The churn prediction component of this project uses a Random Forest classifier to predict customer churn based on various features. The model implementation includes:

- Data preparation and exploration
- Model training with optimized hyperparameters
- Comprehensive evaluation using multiple metrics:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrix visualization
  - ROC and Precision-Recall curves
- Feature importance analysis to identify key drivers of churn
- Detailed explanations of evaluation metrics and model performance assessment

The model is designed to help businesses identify customers at risk of churning, allowing for targeted retention strategies.

## Web Interface

The project includes a web application built with Streamlit that provides:

- A user-friendly interface for entering customer data
- Real-time predictions of churn probability
- Visual representation of prediction results
- Recommendations based on prediction outcome
- Model information and feature importance visualization

The web interface makes the model accessible to non-technical users, allowing them to:

- Input customer information through an organized form
- Receive clear predictions with probability scores
- Understand the factors driving churn through visualizations
- Get actionable recommendations for retention strategies

![Feature Importance](feature_importance.png)

## Technologies Used

- Python
- Pandas for data manipulation
- Scikit-learn for machine learning models
- Jupyter Notebooks for interactive analysis
- Data visualization libraries (Matplotlib, Seaborn)
- Streamlit for the web application
- Pickle for model serialization

## Getting Started

To run this project locally:

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks to see the analysis process
4. Execute the Python scripts for specific feature engineering tasks
5. Run the modeling notebook to train and evaluate the churn prediction model
6. Train and save the model for the web app:
   ```
   python train_model.py
   ```
7. Launch the web application:
   ```
   streamlit run app.py
   ```

## Future Work

- Implement additional machine learning models for comparison
- Optimize feature selection for better model performance
- Expand the analysis to include additional financial metrics
- Add more advanced visualizations to the web interface
- Implement model monitoring and retraining pipeline
