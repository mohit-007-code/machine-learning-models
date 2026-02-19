import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("../model/hog_svm_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# -----------------------------
# HOG Parameters (MUST MATCH TRAINING)
# -----------------------------
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128,128))
    
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )
    
    return features


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐶🐱 Cat vs Dog Classifier (HOG + SVM)")
st.write("Upload an image and let the model predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Extract features
    features = extract_features(image_np)
    
    # Scale
    features_scaled = scaler.transform([features])

    # Predict
    prediction = model.predict(features_scaled)

    # Decision score
    score = model.decision_function(features_scaled)

    if prediction[0] == 0:
        st.success("Prediction: Cat 🐱")
    else:
        st.success("Prediction: Dog 🐶")

    st.write(f"Decision Score: {score[0]:.4f}")
