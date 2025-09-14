import numpy as np
import os
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(__file__)  # Path to Project.py

# Load models with error handling
try:
    with open(os.path.join(BASE_DIR, 'logistic_model.pkl'), 'rb') as file1:
        logistic_model = pickle.load(file1)
except FileNotFoundError:
    st.error("logistic_model.pkl not found! Check if file is uploaded.")

try:
    with open(os.path.join(BASE_DIR, 'knn_model.pkl'), 'rb') as file2:
        knn_model = pickle.load(file2)
except FileNotFoundError:
    st.error("knn_model.pkl not found! Check if file is uploaded.")

try:
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as file3:
        scaler = pickle.load(file3)
except FileNotFoundError:
    st.error("scaler.pkl not found! Check if file is uploaded.")

try:
    with open(os.path.join(BASE_DIR, 'naive_bayes_model.pkl'), 'rb') as file4:
        naive_bayes_model = pickle.load(file4)
except FileNotFoundError:
    st.error("naive_bayes_model.pkl not found! Check if file is uploaded.")

# Streamlit UI
st.title("Heart Disease Prediction")

# Take user input
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.selectbox('Sex (1 = male, 0 = female)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting BP', min_value=80, max_value=200)
chol = st.number_input('Cholesterol', min_value=100, max_value=600)
fbs = st.selectbox('Fasting Blood Sugar > 120 (1 = yes, 0 = no)', [0, 1])
restecg = st.selectbox('RestECG (0-2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate', min_value=60, max_value=220)
exang = st.selectbox('Exercise Induced Angina (1 = yes, 0 = no)', [0, 1])
oldpeak = st.number_input('Oldpeak (ST Depression)', min_value=0.0, max_value=6.0, step=0.1)
slope = st.selectbox('Slope (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thal (1 = normal, 2 = fixed, 3 = reversible)', [1, 2, 3])

# Create input array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input data using the same scaler
scaled_input = scaler.transform(input_data)

# Predict
model_choice = st.selectbox('Choose Model', ['Logistic Regression', 'KNN',
                                             'Naive Bayes'])

if st.button('Predict'):
    if model_choice == 'Logistic Regression':
        prediction = logistic_model.predict(scaled_input)[0]
    elif model_choice == 'KNN':
        prediction = knn_model.predict(scaled_input)[0]
    elif model_choice == 'Naive Bayes':
        prediction = naive_bayes_model.predict(scaled_input)[0]

    if prediction == 1:
        st.error('⚠️ The person has Heart Disease')
    else:
        st.success('✅ The person does NOT have Heart Disease')

st.markdown("---")  # adds a horizontal line
st.markdown("Made with ❤️ by You")
