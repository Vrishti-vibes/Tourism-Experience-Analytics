# Tourism-Experience-Analytics
End-to-end ML system for tourism rating prediction, visit classification &amp; recommendation.

## Project Overview
This project builds a complete end-to-end Machine Learning system for tourism platforms. 
It includes:

- Rating Prediction (Regression)
- Visit Mode Prediction (Classification)
- Attraction Recommendation System

## Problem Statement
Tourism platforms need to enhance user experience using data-driven personalization.

## Objectives

### 1. Regression
Predict attraction rating (1-5 scale)

### 2. Classification
Predict visit mode (Business, Family, Couples, Friends)

### 3. Recommendation
Recommend attractions based on city and ratings

## Dataset Used
- Transaction
- User
- City
- Country
- Region
- Continent
- Attraction
- Attraction Type
- Visit Mode

Total Records: 52,930

## Models Used

### Regression:
- Linear Regression
- Random Forest
- XGBoost

Best Model: XGBoost (R2 = 0.13)

### Classification:
- Logistic Regression
- Random Forest
- XGBoost

Best Model: Random Forest (Accuracy = 51%)

## Business Insights
- Family & Couples give higher ratings
- Seasonal patterns affect satisfaction
- Cultural attractions perform globally strong

## Tech Stack
Python, Pandas, Sklearn, XGBoost, Streamlit

## Future Scope
- Real-time personalization
- GPS-based recommendations
- Deep Learning embeddings
