import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Student Pass Predictor", layout="centered")

st.title("🎓 Student Pass / Fail Predictor")
st.write("Logistic Regression + StandardScaler")

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("../data/student_pass_logistic.csv")

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['pass_exam']

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Model Training
# ---------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🧮 Enter Student Details")

study_hours = st.sidebar.slider("Study Hours (per day)", 1, 10, 5)
attendance = st.sidebar.slider("Attendance (%)", 40, 100, 75)
previous_score = st.sidebar.slider("Previous Exam Score", 30, 100, 60)

# ---------------------------
# Prediction
# ---------------------------
input_data = np.array([[study_hours, attendance, previous_score]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ---------------------------
# Output
# ---------------------------
st.subheader("📊 Prediction Result")

if prediction == 1:
    st.success("✅ Student is likely to PASS")
else:
    st.error("❌ Student is likely to FAIL")

st.write(f"**Probability of Passing:** `{probability:.2f}`")

# ---------------------------
# Explanation
# ---------------------------
st.markdown("---")
st.subheader("ℹ️ How this works")
st.write("""
- Logistic Regression predicts a **probability**
- If probability ≥ 0.5 → PASS
- Otherwise → FAIL
- Inputs are **standardized** before prediction
""")
