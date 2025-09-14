# Heart Disease Prediction

A **Streamlit-based web application** to predict the likelihood of heart disease based on user-provided clinical features. The app uses machine learning models trained on a heart disease dataset to provide predictions in real-time.

---

## Features

The app takes the following inputs from the user:

- **Age**: Number input (1-120)  
- **Sex**: Selectbox (1 = male, 0 = female)  
- **Chest Pain Type (CP)**: Selectbox (0-3)  
- **Resting Blood Pressure (trestbps)**: Number input (80-200)  
- **Cholesterol (chol)**: Number input (100-600)  
- **Fasting Blood Sugar > 120 (fbs)**: Selectbox (0 = no, 1 = yes)  
- **Resting ECG (restecg)**: Selectbox (0-2)  
- **Maximum Heart Rate (thalach)**: Number input (60-220)  
- **Exercise Induced Angina (exang)**: Selectbox (0 = no, 1 = yes)  
- **Oldpeak (ST Depression)**: Number input (0.0-6.0, step=0.1)  
- **Slope**: Selectbox (0-2)  
- **Number of Major Vessels (ca)**: Selectbox (0-3)  
- **Thalassemia (thal)**: Selectbox (1 = normal, 2 = fixed, 3 = reversible)

The app uses the input to predict the **presence or absence of heart disease**.

---

## Machine Learning Algorithms Used

- **Logistic Regression** – Predicts the probability of heart disease.  
- **K-Nearest Neighbors (KNN)** – Classifies patients based on similarity to the training data.  
- **Scaler (StandardScaler/MinMaxScaler)** – Normalizes input features before prediction.  

The model was **trained on a heart disease dataset** and saved for real-time prediction in the Streamlit app.

---

### Installation

1. Clone this repository:

```bash
git clone https://github.com/Prem07a/Heart-Disease.git
```

2. Navigate to the project directory:

```bash
cd Heart-Disease
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```
