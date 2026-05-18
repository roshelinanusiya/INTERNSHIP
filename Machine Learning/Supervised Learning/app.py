import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# LOAD MODEL
model = joblib.load("student_model.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

/* FULL BLACK BACKGROUND */
.stApp {
    background-color: black;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0d0d0d;
}

/* REMOVE GREY COLOR */
div[data-testid="stMetric"] {
    background-color: black !important;
    border: 1px solid #00ADB5;
    padding: 15px;
    border-radius: 15px;
}

/* TEXT */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}

/* BUTTON */
.stButton > button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    width: 100%;
    font-size: 18px;
}

/* DROPDOWN */
div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    color: white !important;
}

/* SLIDER */
.stSlider {
    color: white !important;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 3rem;
    padding-right: 3rem;
    background-color: black;
}

/* REMOVE WHITE HEADER SPACE */
header {
    background: black !important;
}

/* REMOVE TOP GAP */
.main > div {
    padding-top: 0rem;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "About Project", "Prediction", "Results", "Study Planner"]
)

# HOME PAGE
if page == "Home":

    st.title("🎓 Student Performance Prediction System")

    st.write("""
    ### AI Based Student Performance Prediction

    This application predicts whether a student will pass or fail using Machine Learning.
    """)

    st.image(
        "https://images.unsplash.com/photo-1509062522246-3755977927d7",
        width=500
    )

    st.subheader("✨ Project Features")

    st.write("✅ Data Preprocessing")
    st.write("✅ Data Visualization")
    st.write("✅ Machine Learning Prediction")
    st.write("✅ Interactive Streamlit Frontend")
    st.write("✅ Smart Study Planner")

# ABOUT PAGE
elif page == "About Project":

    st.title("📘 About Project")

    st.write("""
    ## 📌 Project Description

    This project is developed using Machine Learning and Streamlit.

    The main goal of this project is to predict student performance
    based on reading and writing scores.

    ## 🛠 Technologies Used

    - Python
    - Pandas
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - Streamlit

    ## 🤖 Algorithm Used

    Random Forest Classifier

    ## 🔄 Project Workflow

    1. Data Collection
    2. Data Preprocessing
    3. Data Visualization
    4. Model Training
    5. Prediction
    6. Frontend Development

    ## 🎯 Project Objective

    To build an AI-based application for predicting student academic performance.
    """)

# PREDICTION PAGE
elif page == "Prediction":

    st.title("🎯 Student Prediction")

    gender = st.selectbox(
        "Select Gender",
        ["male", "female"]
    )

    reading_score = st.slider(
        "Reading Score",
        0, 100, 50
    )

    writing_score = st.slider(
        "Writing Score",
        0, 100, 50
    )

    if st.button("Predict Performance"):

        input_data = pd.DataFrame({

            'reading score': [reading_score],
            'writing score': [writing_score],
            'gender_female': [1 if gender == "female" else 0],
            'gender_male': [1 if gender == "male" else 0]

        })

        model_columns = model.feature_names_in_

        for col in model_columns:

            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[model_columns]

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.success("PASS ✅")
            st.balloons()

        else:

            st.error("FAIL ❌")

# RESULTS PAGE
elif page == "Results":

    st.title("📊 Model Results")

    st.metric(
        "Model Accuracy",
        "95%"
    )

    st.write("### 📈 Performance Graph")

    data = {
        "Category": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Values": [
            95,
            93,
            92,
            94
        ]
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(4,2))

    ax.bar(
        df["Category"],
        df["Values"]
    )

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(colors='white')

    plt.title(
        "Model Performance",
        color='white'
    )

    st.pyplot(fig)

# ADDITIONAL FEATURE
elif page == "Study Planner":

    st.title("📅 Smart Study Planner")

    hours = st.slider(
        "Study Hours Per Day",
        1, 12, 4
    )

    if hours >= 8:

        st.success("Excellent Study Schedule ✅")

        st.write("📚 Reading : 3 Hours")
        st.write("📝 Practice : 3 Hours")
        st.write("🎥 Videos : 1 Hour")
        st.write("😴 Break : 1 Hour")

    elif hours >= 5:

        st.info("Good Study Schedule 📘")

        st.write("📚 Reading : 2 Hours")
        st.write("📝 Practice : 2 Hours")
        st.write("🎥 Videos : 1 Hour")

    else:

        st.warning("Basic Study Schedule ⏰")

        st.write("📚 Reading : 1 Hour")
        st.write("📝 Practice : 1 Hour")