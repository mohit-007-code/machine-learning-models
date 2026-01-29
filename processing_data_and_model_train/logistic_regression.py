# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import joblib


# ===============================
# LOAD SAVED ARTIFACTS
# ===============================
# ✅ model trained earlier
model = joblib.load("big_sale_model.pkl")

# ✅ scaler fitted on training data
scaler = joblib.load("scaler.pkl")

# ✅ FEATURE NAMES (CRITICAL FIX)
# 🔧 NEW: this guarantees inference uses EXACT same columns as training
feature_names = joblib.load("feature_names.pkl")


# ===============================
# STREAMLIT UI
# ===============================
st.title("Big Sale Prediction App")

st.write("Predict whether a sale will be a **Big Sale** or not.")


# ===============================
# USER INPUTS
# ===============================
boxes_shipped = st.number_input("Boxes Shipped", min_value=0)

month = st.selectbox("Month", list(range(1, 13)))

weekday = st.selectbox(
    "Weekday (0 = Monday, 6 = Sunday)",
    list(range(7))
)

country_selected = st.selectbox(
    "Country",
    ["Canada", "India", "UK", "USA", "Australia", "New Zealand"]
)

product_selected = st.selectbox(
    "Product",
    [
        "70% Dark Bites",
        "85% Dark Bars",
        "Milk Chocolate"   # ⚠️ baseline category (no dummy column)
    ]
)


# ===============================
# CREATE INPUT DATA (MOST IMPORTANT PART)
# ===============================

# 🔧 NEW (VERY IMPORTANT):
# Create a ZERO-filled dataframe with ALL training features
# This fixes:
# ❌ missing columns
# ❌ extra columns
# ❌ wrong column order
input_data = pd.DataFrame(0, index=[0], columns=feature_names)


# ===============================
# FILL NUMERIC FEATURES
# ===============================
input_data["Boxes Shipped"] = boxes_shipped
input_data["Month"] = month
input_data["Weekday"] = weekday


# ===============================
# HANDLE COUNTRY (FIXED)
# ===============================
# 🔧 CHANGE:
# We ONLY set the dummy column IF it exists
# Because with drop_first=True, one country is baseline (all zeros)
country_col = f"Country_{country_selected}"

if country_col in input_data.columns:
    input_data[country_col] = 1
# else:
# baseline country → leave all zeros


# ===============================
# HANDLE PRODUCT (FIXED)
# ===============================
# 🔧 CHANGE:
# Same logic as Country
# Milk Chocolate was dropped during training (baseline)
product_col = f"Product_{product_selected}"

if product_col in input_data.columns:
    input_data[product_col] = 1
# else:
# baseline product → leave all zeros


# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):
    # 🔧 SAME scaler used during training
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Big Sale (Probability: {probability:.2f})")
    else:
        st.warning(f"❌ Not a Big Sale (Probability: {probability:.2f})")


# ===============================
# DEBUG (OPTIONAL – REMOVE LATER)
# ===============================
# st.write("Input Data:", input_data)
# st.write("Feature Names:", feature_names)
