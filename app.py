
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load saved model
@st.cache_resource
def load_model():
    model = joblib.load("student_score_model.joblib")
    features = joblib.load("feature_names.joblib")
    return model, features

try:
    model, features = load_model()
except FileNotFoundError:
    st.error(
        "Model files are missing. Place both .joblib files "
        "in the same folder as app.py."
    )
    st.stop()

# App title
st.title("🎓 Student Performance Predictor")
st.write(
    "Enter student details below to estimate an exam score "
    "using a trained machine learning model."
)

st.info(
    "Educational demo using synthetic data. "
    "Predictions are illustrative, not real student assessments."
)

# Interactive inputs
st.sidebar.header("Student Details")

study_hours = st.sidebar.slider(
    "Study hours per day", 0.0, 10.0, 4.0, 0.5
)
attendance = st.sidebar.slider(
    "Attendance (%)", 50, 100, 85
)
previous_score = st.sidebar.slider(
    "Previous exam score", 30, 100, 70
)
sleep_hours = st.sidebar.slider(
    "Sleep hours", 4.0, 10.0, 7.0, 0.5
)
assignments_completed = st.sidebar.slider(
    "Assignments completed", 0, 10, 7
)

# Prepare input in the same feature order used for training
input_data = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "assignments_completed": assignments_completed
}])[features]

# Predict
prediction = float(model.predict(input_data)[0])
prediction = max(0.0, min(100.0, prediction))

# Display result
st.subheader("Predicted Exam Score")
st.metric("Estimated Score", f"{prediction:.1f} / 100")
st.progress(int(round(prediction)))

if prediction >= 80:
    st.success("The model estimates a high score.")
elif prediction >= 60:
    st.warning("The model estimates a moderate score.")
else:
    st.error("The model estimates a lower score.")

with st.expander("View submitted inputs"):
    st.dataframe(input_data, use_container_width=True)

with st.expander("About this project"):
    st.write(
        "This app uses a Random Forest regression model trained "
        "on synthetic sample data. Its predictions are for "
        "demonstration purposes only."
    )
