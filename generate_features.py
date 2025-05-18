import pandas as pd
import numpy as np
from datetime import datetime

print("Loading data...")
# Load the cleaned data
df = pd.read_csv('./clean_data_after_eda.csv')

# Convert date columns to datetime
df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Number of customers: {df['id'].nunique()}")
print(f"Churn rate: {df['churn'].mean():.2%}")

print("\nGenerating time-based features...")
# Create a reference date (end of the dataset period)
reference_date = df['date_end'].max()
print(f"Reference date: {reference_date}")

# Contract duration in days
df['contract_duration_days'] = (df['date_end'] - df['date_activ']).dt.days

# Time since last product modification in days
df['days_since_last_modification'] = (reference_date - df['date_modif_prod']).dt.days

# Time until renewal in days (negative values mean renewal is in the past)
df['days_until_renewal'] = (df['date_renewal'] - reference_date).dt.days

# Extract month and quarter for seasonality
df['activation_month'] = df['date_activ'].dt.month
df['activation_quarter'] = df['date_activ'].dt.quarter
df['renewal_month'] = df['date_renewal'].dt.month
df['renewal_quarter'] = df['date_renewal'].dt.quarter

# Calculate if product was modified recently (within last 6 months)
six_months = 180  # approximately 6 months in days
df['recent_modification'] = (df['days_since_last_modification'] <= six_months).astype(int)

# Calculate if contract is near expiration (within next 3 months)
three_months = 90  # approximately 3 months in days
df['near_expiration'] = ((df['days_until_renewal'] >= 0) & (df['days_until_renewal'] <= three_months)).astype(int)

print("\nGenerating consumption-based features...")
# Calculate consumption per day (to normalize for different contract durations)
df['cons_per_day'] = df['cons_12m'] / 365  # Average daily consumption over the year

# Calculate gas consumption per day (if applicable)
df['gas_cons_per_day'] = df['cons_gas_12m'] / 365  # Average daily gas consumption

# Calculate ratio of last month consumption to average monthly consumption
df['last_month_to_avg_ratio'] = df['cons_last_month'] / (df['cons_12m'] / 12)
# Replace infinity and NaN with 0
df['last_month_to_avg_ratio'] = df['last_month_to_avg_ratio'].replace([np.inf, -np.inf, np.nan], 0)

# Calculate difference between forecasted and actual consumption
df['forecast_vs_actual_diff'] = df['forecast_cons_12m'] - df['cons_12m']
df['forecast_vs_actual_ratio'] = df['forecast_cons_12m'] / df['cons_12m']
# Replace infinity and NaN with 0
df['forecast_vs_actual_ratio'] = df['forecast_vs_actual_ratio'].replace([np.inf, -np.inf, np.nan], 0)

# Calculate if customer has both electricity and gas
df['has_both_services'] = ((df['cons_12m'] > 0) & (df['cons_gas_12m'] > 0)).astype(int)

# Calculate if customer has zero consumption (inactive)
df['is_inactive'] = ((df['cons_12m'] == 0) & (df['cons_last_month'] == 0)).astype(int)

# Calculate if customer has high consumption variability
df['high_consumption_variability'] = (df['last_month_to_avg_ratio'] > 1.5).astype(int)

print("\nGenerating price-based features...")
# Calculate price differences between peak and off-peak
df['peak_offpeak_price_diff'] = df['forecast_price_energy_peak'] - df['forecast_price_energy_off_peak']

# Calculate price ratios
df['peak_offpeak_price_ratio'] = df['forecast_price_energy_peak'] / df['forecast_price_energy_off_peak']
# Replace infinity and NaN with 0
df['peak_offpeak_price_ratio'] = df['peak_offpeak_price_ratio'].replace([np.inf, -np.inf, np.nan], 0)

# Calculate yearly price variation metrics
df['year_price_var_total'] = df['var_year_price_off_peak_var'] + df['var_year_price_peak_var'] + df['var_year_price_mid_peak_var']
df['year_price_fix_total'] = df['var_year_price_off_peak_fix'] + df['var_year_price_peak_fix'] + df['var_year_price_mid_peak_fix']
df['year_price_total'] = df['var_year_price_off_peak'] + df['var_year_price_peak'] + df['var_year_price_mid_peak']

# Calculate 6-month price variation metrics
df['6m_price_var_total'] = df['var_6m_price_off_peak_var'] + df['var_6m_price_peak_var'] + df['var_6m_price_mid_peak_var']
df['6m_price_fix_total'] = df['var_6m_price_off_peak_fix'] + df['var_6m_price_peak_fix'] + df['var_6m_price_mid_peak_fix']
df['6m_price_total'] = df['var_6m_price_off_peak'] + df['var_6m_price_peak'] + df['var_6m_price_mid_peak']

# Calculate price change acceleration (difference between 6-month and yearly changes)
df['price_change_acceleration'] = df['6m_price_total'] - df['year_price_total']

# Calculate if customer has experienced a significant price increase
df['significant_price_increase'] = (df['year_price_total'] > 0.05).astype(int)  # 5% threshold

print("\nGenerating customer profile features...")
# Calculate customer lifetime value (simplified as net_margin * tenure)
df['customer_lifetime_value'] = df['net_margin'] * df['num_years_antig']

# Calculate margin per consumption unit
df['margin_per_consumption'] = df['net_margin'] / df['cons_12m']
# Replace infinity and NaN with 0
df['margin_per_consumption'] = df['margin_per_consumption'].replace([np.inf, -np.inf, np.nan], 0)

# Calculate if customer has multiple products
df['has_multiple_products'] = (df['nb_prod_act'] > 1).astype(int)

# Calculate power utilization ratio (actual consumption vs. maximum power)
df['power_utilization_ratio'] = df['cons_12m'] / (df['pow_max'] * 24 * 365)  # Simplified calculation
# Replace infinity and NaN with 0
df['power_utilization_ratio'] = df['power_utilization_ratio'].replace([np.inf, -np.inf, np.nan], 0)

# Calculate if customer is high-value
df['is_high_value'] = (df['net_margin'] > df['net_margin'].quantile(0.75)).astype(int)

# Calculate if customer is long-term
df['is_long_term'] = (df['num_years_antig'] >= 5).astype(int)

# Calculate if customer has high power needs
df['has_high_power'] = (df['pow_max'] > df['pow_max'].quantile(0.75)).astype(int)

print("\nGenerating interaction features...")
# Create interaction features between different feature groups

# Time x Consumption interactions
df['tenure_consumption_interaction'] = df['num_years_antig'] * df['cons_per_day']
df['recent_mod_consumption_change'] = df['recent_modification'] * df['last_month_to_avg_ratio']

# Price x Consumption interactions
df['price_sensitivity'] = df['year_price_total'] * df['cons_per_day']
df['price_forecast_interaction'] = df['significant_price_increase'] * df['forecast_vs_actual_ratio']

# Customer profile x Time interactions
df['value_tenure_interaction'] = df['is_high_value'] * df['is_long_term']
df['expiration_value_interaction'] = df['near_expiration'] * df['is_high_value']

# Create the feature for the difference between off-peak prices in December and January
# (similar to the example in the starter notebook)
print("\nGenerating price difference features (Dec-Jan)...")
try:
    # Try to load the price data if it exists
    price_df = pd.read_csv('price_data.csv')
    price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')
    
    # Group off-peak prices by companies and month
    monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({
        'price_off_peak_var': 'mean', 
        'price_off_peak_fix': 'mean'
    }).reset_index()
    
    # Get january and december prices
    jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
    dec_prices = monthly_price_by_id.groupby('id').last().reset_index()
    
    # Calculate the difference
    diff = pd.merge(
        dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), 
        jan_prices.drop(columns='price_date'), 
        on='id'
    )
    diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
    diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
    diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
    
    # Merge with main dataframe
    df = pd.merge(df, diff, on='id', how='left')
    df['offpeak_diff_dec_january_energy'] = df['offpeak_diff_dec_january_energy'].fillna(0)
    df['offpeak_diff_dec_january_power'] = df['offpeak_diff_dec_january_power'].fillna(0)
    
    print("Successfully added price difference features.")
except Exception as e:
    print(f"Could not generate price difference features: {e}")
    # Create dummy features if price data is not available
    df['offpeak_diff_dec_january_energy'] = 0
    df['offpeak_diff_dec_january_power'] = 0

# List all engineered features
time_features = ['contract_duration_days', 'days_since_last_modification', 'days_until_renewal', 
                 'activation_month', 'activation_quarter', 'renewal_month', 'renewal_quarter',
                 'recent_modification', 'near_expiration']

consumption_features = ['cons_per_day', 'gas_cons_per_day', 'last_month_to_avg_ratio',
                        'forecast_vs_actual_diff', 'forecast_vs_actual_ratio',
                        'has_both_services', 'is_inactive', 'high_consumption_variability']

price_features = ['peak_offpeak_price_diff', 'peak_offpeak_price_ratio',
                  'year_price_var_total', 'year_price_fix_total', 'year_price_total',
                  '6m_price_var_total', '6m_price_fix_total', '6m_price_total',
                  'price_change_acceleration', 'significant_price_increase',
                  'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']

profile_features = ['customer_lifetime_value', 'margin_per_consumption',
                    'has_multiple_products', 'power_utilization_ratio',
                    'is_high_value', 'is_long_term', 'has_high_power']

interaction_features = ['tenure_consumption_interaction', 'recent_mod_consumption_change',
                        'price_sensitivity', 'price_forecast_interaction',
                        'value_tenure_interaction', 'expiration_value_interaction']

all_engineered_features = time_features + consumption_features + price_features + profile_features + interaction_features

# Save the engineered features
print("\nSaving engineered features...")
columns_to_save = ['id', 'churn'] + all_engineered_features
df[columns_to_save].to_csv('engineered_features.csv', index=False)

print(f"Saved {len(all_engineered_features)} engineered features to 'engineered_features.csv'")

# Print a sample of the engineered features
print("\nSample of engineered features:")
print(df[columns_to_save].head(3))

print("\nFeature engineering complete!")
