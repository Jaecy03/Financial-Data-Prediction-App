# Feature Engineering Summary

## Overview
This document summarizes the engineered features created for the customer churn prediction model. A total of 42 new features were created across several categories.

## Feature Categories

### 1. Time-based Features
These features capture temporal aspects of customer relationships and contracts.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `contract_duration_days` | Number of days between activation and end date | Longer contracts may indicate stronger customer commitment |
| `days_since_last_modification` | Days since the last product modification | Recent modifications might affect churn risk |
| `days_until_renewal` | Days until contract renewal | Customers near renewal are at higher risk of churn |
| `activation_month` | Month of contract activation | Captures seasonal patterns in customer acquisition |
| `activation_quarter` | Quarter of contract activation | Broader seasonal patterns |
| `renewal_month` | Month of contract renewal | Captures seasonal patterns in renewals |
| `renewal_quarter` | Quarter of contract renewal | Broader seasonal patterns in renewals |
| `recent_modification` | Flag for modifications in last 6 months | Recent changes may increase churn risk |
| `near_expiration` | Flag for contracts expiring within 3 months | Imminent expiration increases churn risk |

### 2. Consumption-based Features
These features analyze customer usage patterns and consumption behavior.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `cons_per_day` | Average daily electricity consumption | Normalizes consumption across different contract lengths |
| `gas_cons_per_day` | Average daily gas consumption | Normalizes gas consumption |
| `last_month_to_avg_ratio` | Ratio of last month's consumption to 12-month average | Identifies recent changes in consumption patterns |
| `forecast_vs_actual_diff` | Difference between forecasted and actual consumption | Measures forecast accuracy and consumption predictability |
| `forecast_vs_actual_ratio` | Ratio of forecasted to actual consumption | Alternative measure of forecast accuracy |
| `has_both_services` | Flag for customers with both electricity and gas | Multi-service customers may be less likely to churn |
| `is_inactive` | Flag for customers with zero consumption | Inactive customers may be at high risk of churn |
| `high_consumption_variability` | Flag for customers with highly variable consumption | Variability may indicate changing needs or dissatisfaction |

### 3. Price-based Features
These features examine pricing structures, variations, and customer sensitivity to price changes.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `peak_offpeak_price_diff` | Difference between peak and off-peak prices | Measures price spread |
| `peak_offpeak_price_ratio` | Ratio of peak to off-peak prices | Alternative measure of price spread |
| `year_price_var_total` | Sum of yearly variable price changes | Captures overall variable price changes |
| `year_price_fix_total` | Sum of yearly fixed price changes | Captures overall fixed price changes |
| `year_price_total` | Sum of all yearly price changes | Captures total price impact |
| `6m_price_var_total` | Sum of 6-month variable price changes | Captures recent variable price changes |
| `6m_price_fix_total` | Sum of 6-month fixed price changes | Captures recent fixed price changes |
| `6m_price_total` | Sum of all 6-month price changes | Captures total recent price impact |
| `price_change_acceleration` | Difference between 6-month and yearly changes | Measures if price changes are accelerating |
| `significant_price_increase` | Flag for price increases > 5% | Identifies customers experiencing notable price hikes |
| `offpeak_diff_dec_january_energy` | Difference in off-peak energy prices between December and January | Captures seasonal price variations |
| `offpeak_diff_dec_january_power` | Difference in off-peak power prices between December and January | Captures seasonal price variations |

### 4. Customer Profile Features
These features characterize customer value, behavior, and relationship with the company.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `customer_lifetime_value` | Net margin multiplied by tenure | Estimates overall customer value |
| `margin_per_consumption` | Net margin divided by consumption | Measures profitability per unit consumed |
| `has_multiple_products` | Flag for customers with multiple products | Multi-product customers may be less likely to churn |
| `power_utilization_ratio` | Ratio of consumption to maximum possible | Measures how efficiently customers use their power capacity |
| `is_high_value` | Flag for customers in top 25% by net margin | Identifies high-value customers |
| `is_long_term` | Flag for customers with 5+ years tenure | Identifies long-term customers |
| `has_high_power` | Flag for customers in top 25% by maximum power | Identifies customers with high power needs |

### 5. Interaction Features
These features capture relationships between different feature categories.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `tenure_consumption_interaction` | Tenure multiplied by consumption per day | Captures relationship between loyalty and usage |
| `recent_mod_consumption_change` | Recent modification flag multiplied by consumption ratio | Identifies if recent changes affected consumption |
| `price_sensitivity` | Price changes multiplied by consumption | Measures potential impact of price changes on high-usage customers |
| `price_forecast_interaction` | Price increase flag multiplied by forecast ratio | Identifies price-sensitive customers with changing consumption |
| `value_tenure_interaction` | High-value flag multiplied by long-term flag | Identifies valuable long-term customers |
| `expiration_value_interaction` | Near expiration flag multiplied by high-value flag | Identifies valuable customers at risk of churning soon |

## Potential Applications

These engineered features can be used to:

1. **Improve churn prediction models** by capturing complex relationships in the data
2. **Identify high-risk customer segments** for targeted retention campaigns
3. **Understand key drivers of churn** across different customer types
4. **Develop personalized retention strategies** based on specific risk factors
5. **Optimize pricing strategies** to minimize churn risk while maximizing revenue

## Next Steps

1. Evaluate feature importance in predictive models
2. Refine features based on model performance
3. Consider additional feature engineering opportunities:
   - Text analysis of customer communications
   - Geographic/regional features
   - Competitor pricing and market conditions
   - Customer service interactions
   - Payment history and billing features
