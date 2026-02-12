import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../data/student_performance_lr.csv")

X = df[['study_hours','attendance_percent','previous_score',
        'sleep_hours','social_media_hours']]
y = df['final_score']

# -----------------------------
# Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎓 Student Performance Prediction")
st.write("Predict final exam score using Linear Regression")

study_hours = st.slider("Study Hours per Day", 1.0, 8.0, 4.0)
attendance = st.slider("Attendance (%)", 60, 100, 85)
prev_score = st.slider("Previous Exam Score", 40, 95, 75)
sleep = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
social_media = st.slider("Social Media Usage (hrs/day)", 0.0, 6.0, 2.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Score"):
    new_data = np.array([[study_hours, attendance,
                           prev_score, sleep, social_media]])

    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    st.success(f"📊 Predicted Final Score: {prediction[0]:.2f}")
