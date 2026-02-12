import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../model/svm_course_recommender.pkl")

model = joblib.load(model_path)

st.title("🎓 Student Course Recommendation System")
st.write("Enter student details below:")

math = st.slider("Math Score", 0, 100, 70)
programming = st.slider("Programming Score", 0, 100, 70)
logical = st.slider("Logical Reasoning Score", 0, 100, 70)
communication = st.slider("Communication Skill Score", 0, 100, 70)
gpa = st.slider("GPA", 0.0, 10.0, 7.0)

interest_ai = st.selectbox("Interest in AI", ["Yes", "No"])
interest_web = st.selectbox("Interest in Web", ["Yes", "No"])
interest_cyber = st.selectbox("Interest in Cyber", ["Yes", "No"])
interest_cloud = st.selectbox("Interest in Cloud", ["Yes", "No"])

interest_ai = 1 if interest_ai == "Yes" else 0
interest_web = 1 if interest_web == "Yes" else 0
interest_cyber = 1 if interest_cyber == "Yes" else 0
interest_cloud = 1 if interest_cloud == "Yes" else 0

if st.button("Recommend Course"):

    input_data = pd.DataFrame([{
        "Math_Score": math,
        "Programming_Score": programming,
        "Logical_Reasoning_Score": logical,
        "Communication_Skill_Score": communication,
        "GPA": gpa,
        "Interest_AI": interest_ai,
        "Interest_Web": interest_web,
        "Interest_Cyber": interest_cyber,
        "Interest_Cloud": interest_cloud
    }])

    prediction = model.predict(input_data)[0]

    proba = model.predict_proba(input_data)[0]
    classes = model.named_steps["model"].classes_

    top3_idx = np.argsort(proba)[-3:][::-1]

    st.success(f"🎯 Recommended Course: {prediction}")
    st.subheader("Top 3 Suggestions")

    for idx in top3_idx:
        st.write(f"{classes[idx]} : {round(proba[idx]*100, 2)} %")
