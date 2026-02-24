import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODELS ----------------
rating_model = joblib.load("rating_model.pkl")
visit_mode_model = joblib.load("visit_mode_model.pkl")
le_target = joblib.load("visitmode_encoder.pkl")
attraction_matrix = joblib.load("attraction_matrix.pkl")

st.title("Tourism Experience Recommender")

# ---------------- USER INPUT ----------------
TransactionId = st.number_input("Transaction ID", min_value=1, value=1)
UserId = st.number_input("User ID", min_value=1, value=1)
VisitYear = st.number_input("Visit Year", min_value=2000, max_value=2030, value=2026)
VisitMonth = st.number_input("Visit Month", min_value=1, max_value=12, value=1)

VisitModeId = st.number_input("Visit Mode ID", min_value=0, value=0)

AttractionId = st.number_input("Attraction ID", min_value=1, value=1)
ContinentId = st.number_input("Continent ID", min_value=1, value=1)
RegionId = st.number_input("Region ID", min_value=1, value=1)
CountryId = st.number_input("Country ID", min_value=1, value=1)
CityId = st.number_input("City ID", min_value=1, value=1)

# ---------------- BUILD RATING INPUT ----------------
rating_input = pd.DataFrame([{
    'TransactionId': TransactionId,
    'UserId': UserId,
    'VisitYear': VisitYear,
    'VisitMonth': VisitMonth,
    'VisitModeId': VisitModeId,
    'AttractionId': AttractionId,
    'ContinentId': ContinentId,
    'RegionId': RegionId,
    'CountryId': CountryId,
    'CityId': CityId
}])

# Ensure correct order & avoid error
rating_input = rating_input.reindex(
    columns=rating_model.feature_names_in_,
    fill_value=0
)

# ---------------- PREDICT RATING ----------------
pred_rating = rating_model.predict(rating_input)[0]
st.success(f"Predicted Rating: {round(pred_rating, 2)}")

# ---------------- BUILD VISIT MODE INPUT ----------------
visit_input = pd.DataFrame([{
    'TransactionId': TransactionId,
    'UserId': UserId,
    'VisitYear': VisitYear,
    'VisitMonth': VisitMonth,
    'AttractionId': AttractionId,
    'Rating': pred_rating
}])

visit_input = visit_input.reindex(
    columns=visit_mode_model.feature_names_in_,
    fill_value=0
)

# ---------------- PREDICT VISIT MODE ----------------
pred_visit_encoded = visit_mode_model.predict(visit_input)[0]
pred_visit_mode = le_target.inverse_transform([pred_visit_encoded])[0]

st.success(f"Predicted Visit Mode: {pred_visit_mode}")

# ---------------- RECOMMENDATIONS ----------------
recommended = attraction_matrix.sort_values(
    by="Rating", ascending=False
).head(5)

st.subheader("Top Recommended Attractions:")
st.dataframe(recommended)