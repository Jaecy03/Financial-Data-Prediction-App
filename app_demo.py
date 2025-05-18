import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load the model and feature names
def load_model():
    """Load the trained model and feature names."""
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, feature_names

# Create a sample prediction
def make_sample_prediction():
    """Make a sample prediction to demonstrate the model."""
    model, feature_names = load_model()
    
    # Create a sample input (all zeros)
    sample_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    
    # Make prediction
    prediction = model.predict(sample_input)[0]
    prediction_proba = model.predict_proba(sample_input)[0][1]
    
    print(f"Sample Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    print(f"Churn Probability: {prediction_proba:.2%}")
    
    # Create feature importance visualization
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(10)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance visualization saved as 'feature_importance.png'")
    
    return prediction, prediction_proba, importance_df

if __name__ == "__main__":
    print("Demonstrating the churn prediction model...")
    prediction, probability, importance = make_sample_prediction()
    print("\nTop 5 most important features:")
    print(importance.head(5))
