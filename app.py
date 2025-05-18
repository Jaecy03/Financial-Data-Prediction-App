"""
Streamlit web app for churn prediction using the trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="��",
    layout="wide"
)

# Function to load the model and feature names
@st.cache_resource
def load_model():
    """Load the trained model and feature names."""
    model_path = 'models/random_forest_model.pkl'
    feature_names_path = 'models/feature_names.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
        st.error("Model files not found. Please run train_model.py first.")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, feature_names

# Load the model and feature names
model, feature_names = load_model()

# Main app
def main():
    """Main function for the Streamlit app."""
    st.title("Customer Churn Prediction")
    st.write("This app predicts whether a customer will churn based on various features.")
    
    # Check if model is loaded
    if model is None or feature_names is None:
        st.warning("Please run train_model.py to train and save the model first.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Make Prediction", "Model Information"])
    
    with tab1:
        st.header("Enter Customer Information")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store input values
        input_data = {}
        
        # Group features into categories for better organization
        consumption_features = [f for f in feature_names if 'cons' in f]
        forecast_features = [f for f in feature_names if 'forecast' in f]
        price_features = [f for f in feature_names if 'price' in f]
        margin_features = [f for f in feature_names if 'margin' in f]
        time_features = [f for f in feature_names if 'months' in f or 'tenure' in f]
        channel_features = [f for f in feature_names if 'channel' in f]
        origin_features = [f for f in feature_names if 'origin' in f]
        other_features = [f for f in feature_names if f not in consumption_features + forecast_features + 
                         price_features + margin_features + time_features + channel_features + origin_features]
        
        # Create a form for user input
        with st.form("prediction_form"):
            # Consumption features
            st.subheader("Consumption Information")
            for i, feature in enumerate(consumption_features):
                col = [col1, col2, col3][i % 3]
                with col:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")
            
            # Forecast features
            st.subheader("Forecast Information")
            for i, feature in enumerate(forecast_features):
                col = [col1, col2, col3][i % 3]
                with col:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")
            
            # Price features
            st.subheader("Price Information")
            for i, feature in enumerate(price_features[:6]):  # Limit to first 6 for brevity
                col = [col1, col2, col3][i % 3]
                with col:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")
            
            with st.expander("More Price Features"):
                for i, feature in enumerate(price_features[6:]):
                    col = [col1, col2, col3][i % 3]
                    with col:
                        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")
            
            # Margin features
            st.subheader("Margin Information")
            for i, feature in enumerate(margin_features):
                col = [col1, col2, col3][i % 3]
                with col:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")
            
            # Time features
            st.subheader("Time Information")
            for i, feature in enumerate(time_features):
                col = [col1, col2, col3][i % 3]
                with col:
                    input_data[feature] = st.number_input(f"{feature}", value=0, step=1)
            
            # Channel and Origin features (as binary)
            st.subheader("Channel and Origin Information")
            for i, feature in enumerate(channel_features + origin_features):
                col = [col1, col2, col3][i % 3]
                with col:
                    input_data[feature] = st.selectbox(f"{feature}", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            # Other features
            if other_features:
                st.subheader("Other Information")
                for i, feature in enumerate(other_features):
                    col = [col1, col2, col3][i % 3]
                    with col:
                        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")
            
            # Submit button
            submitted = st.form_submit_button("Predict Churn")
        
        # Make prediction when form is submitted
        if submitted:
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure the order of columns matches the model's expected features
            input_df = input_df[feature_names]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][1]
            
            # Display prediction
            st.header("Prediction Result")
            
            # Create columns for prediction display
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ Customer is likely to churn")
                else:
                    st.success("✅ Customer is likely to stay")
            
            with col2:
                st.metric("Churn Probability", f"{prediction_proba:.2%}")
            
            # Display gauge chart for probability
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Create a gauge-like visualization
            ax.barh([0], [prediction_proba], color='red', height=0.4)
            ax.barh([0], [1-prediction_proba], left=[prediction_proba], color='green', height=0.4)
            
            # Add labels
            ax.text(0.05, 0, "Low Risk", ha='center', va='center', color='white', fontweight='bold')
            ax.text(0.95, 0, "High Risk", ha='center', va='center', color='white', fontweight='bold')
            ax.text(prediction_proba, -0.5, f"{prediction_proba:.2%}", ha='center', va='center', fontweight='bold')
            
            # Remove axes
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, 1)
            ax.axis('off')
            
            st.pyplot(fig)
            
            # Recommendation based on prediction
            st.subheader("Recommendation")
            if prediction == 1:
                st.write("""
                This customer has a high risk of churning. Consider implementing retention strategies such as:
                - Personalized offers or discounts
                - Proactive customer service outreach
                - Loyalty program enrollment
                - Service upgrades or improvements
                """)
            else:
                st.write("""
                This customer has a low risk of churning. Consider:
                - Continuing to monitor their satisfaction
                - Offering cross-sell or upsell opportunities
                - Encouraging referrals
                - Gathering feedback for service improvements
                """)
    
    with tab2:
        st.header("Model Information")
        st.write("This prediction is powered by a Random Forest classifier trained on historical customer data.")
        
        # Display model parameters
        st.subheader("Model Parameters")
        st.write(f"- Number of trees: 100")
        st.write(f"- Maximum depth: 15")
        st.write(f"- Minimum samples split: 10")
        st.write(f"- Minimum samples leaf: 4")
        st.write(f"- Class weight: Balanced")
        
        # Display feature importance
        st.subheader("Top 10 Feature Importance")
        
        # Get feature importance
        feature_importance = model.feature_importances_
        
        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(10)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Add explanation of what feature importance means
        st.write("""
        **Feature Importance Explanation:**
        
        The chart above shows the top 10 most influential features in predicting customer churn. 
        Features with higher importance have a greater impact on the model's predictions.
        
        Understanding these key factors can help businesses focus their retention efforts on the 
        most critical aspects of the customer relationship.
        """)

if __name__ == "__main__":
    main()
