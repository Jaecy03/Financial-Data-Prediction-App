import pandas as pd
import numpy as np

print("Loading engineered features...")
# Load the engineered features
df = pd.read_csv('engineered_features.csv')

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {df.shape[1] - 2}")  # Subtract id and churn
print(f"Churn rate: {df['churn'].mean():.2%}")

# Calculate correlation with churn
correlations = []
for col in df.columns:
    if col not in ['id', 'churn']:
        corr = df[col].corr(df['churn'])
        correlations.append((col, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nTop 15 Features by Correlation with Churn:")
for i, (feature, corr) in enumerate(correlations[:15]):
    print(f"{i+1}. {feature}: {corr:.4f}")

# Calculate mean values for churned vs non-churned customers
print("\nFeature Means for Churned vs Non-Churned Customers:")
print("Feature | Non-Churned Mean | Churned Mean | Ratio")
print("--------|-----------------|-------------|------")

for feature, _ in correlations[:15]:
    non_churned_mean = df[df['churn'] == 0][feature].mean()
    churned_mean = df[df['churn'] == 1][feature].mean()
    
    # Handle division by zero
    if non_churned_mean == 0:
        ratio = float('inf') if churned_mean > 0 else 1.0
    else:
        ratio = churned_mean / non_churned_mean
    
    print(f"{feature} | {non_churned_mean:.4f} | {churned_mean:.4f} | {ratio:.4f}")

# Save the correlation analysis
correlation_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
correlation_df.to_csv('feature_correlations.csv', index=False)
print("\nFeature correlations saved to 'feature_correlations.csv'")

print("\nFeature analysis complete!")
