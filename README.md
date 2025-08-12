# Airline Delay Prediction

This project predicts whether a flight will be **on-time** or **delayed** by more than 15 minutes using historical U.S. flight data and Machine Learning.

---

## Project Overview
Flight delays can cause major inconvenience for passengers and airlines.  
This project leverages machine learning to predict flight delays based on features like **airline, day of week, departure time, and distance**.

**Key Highlights:**
- Built using **Python, Pandas, Scikit-learn, and XGBoost**
- Deployed as an **interactive Streamlit web app**
- Achieved **97.45% accuracy** on the test dataset

---

##  Dataset
- **Source:** U.S. DOT On-Time Flight Performance Data
- **Target Variable:** `Delayed` (1 = delayed > 15 mins, 0 = on time)
- **Size:** ~2 million records (sampled for training/testing)
- **Key Features:**
  - `UniqueCarrier` – Airline code
  - `DayOfWeek` – Day of the week (1–7)
  - `DepTime` – Departure time (HHMM)
  - `Distance` – Flight distance in miles

---

## ⚙️ Modeling Approach
1. **Data Cleaning & Preprocessing**
   - Removed null values & irrelevant columns
   - Added `Delayed` column based on `ArrDelay` > 15 min
2. **Feature Engineering**
   - One-hot encoding for categorical variables
3. **Model Selection**
   - Compared Logistic Regression, Random Forest, Decision Tree, and XGBoost
   - **XGBoost** chosen for best performance
4. **Evaluation Metrics**
   - **Accuracy:** 97.45%
   - **Precision:** 97%
   - **Recall:** 97%
   - **F1-Score:** 97%

---

##  How to Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/airline-delay-prediction.git
cd airline-delay-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
