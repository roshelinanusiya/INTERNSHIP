import streamlit as st
import pandas as pd
import joblib

# Load the saved model and encoders
model = joblib.load('titanic_model.joblib')
le_sex = joblib.load('le_sex.joblib')
le_embarked = joblib.load('le_embarked.joblib')

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived.")

# User Inputs
pclass = st.selectbox("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation (C, Q, S)", ["C", "Q", "S"])

if st.button("Predict"):
    # Prepare data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Apply the same encoding used in training
    input_data['Sex'] = le_sex.transform(input_data['Sex'])
    input_data['Embarked'] = le_embarked.transform(input_data['Embarked'])

    # Make Prediction
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Result: This passenger likely SURVIVED!")
    else:
        st.error("Result: This passenger likely DID NOT survive.")
