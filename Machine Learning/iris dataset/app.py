import streamlit as st
import joblib
import numpy as np
import base64
from sklearn.datasets import load_iris

# Load model
model = joblib.load("iris_svm_model.pkl")

# Load iris names
iris = load_iris()

# Background image function
def add_bg_from_local(image_file):

    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
        }}

        h1 {{
            color: white;
            text-align: center;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
add_bg_from_local("flower.jpg")

# Title
st.title("🌸 Iris Flower Prediction using SVM")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediction
if st.button("Predict"):

    data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = model.predict(data)

    result = iris.target_names[prediction[0]]

    st.success(f"Predicted Flower: {result}")
