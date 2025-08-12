import streamlit as st
import pickle
import pandas as pd

# =========================
# Load Model & Training Columns
# =========================
with open("xgboost_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("train_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Airline Delay Prediction", page_icon="✈️")

# =========================
# Title & Description
# =========================
st.title("✈️ Airline Delay Prediction")
st.markdown("""
Predict whether a flight will be **on-time** or **delayed** by more than 15 minutes,  
based on historical U.S. flight data using Machine Learning (XGBoost).
""")

# =========================
# User Inputs
# =========================
st.header("Enter Flight Details")

# Example options — change to match your training dataset's actual values
airline = st.selectbox("Airline", ["AA", "UA", "DL", "WN", "B6"])
day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
dep_time = st.number_input("Departure Time (HHMM)", min_value=0, max_value=2359, step=1)
distance = st.number_input("Distance (miles)", min_value=0, max_value=5000, step=1)

# =========================
# Prediction
# =========================
if st.button("Predict Delay"):
    # Create dataframe with same columns as training
    input_df = pd.DataFrame([[airline, day_of_week, dep_time, distance]],
                            columns=["UniqueCarrier", "DayOfWeek", "DepTime", "Distance"])
    
    # One-hot encode & align columns
    input_encoded = pd.get_dummies(input_df).reindex(columns=train_columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]
    
    # Display result
    if prediction == 1:
        st.error(f"Flight likely **DELAYED**! (Probability: {prob:.2%})")
    else:
        st.success(f" Flight likely **ON-TIME** (Probability: {prob:.2%})")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Dataset: U.S. DOT Flight Delay Data | Created by Surajit Nayak")

