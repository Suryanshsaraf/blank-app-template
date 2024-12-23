import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Load dataset
data = pd.read_csv('https://github.com/Suryanshsaraf/blank-app-template/blob/a15e8c6b406c0f92c98c0ed9fa724621d74ec34e/updated_dataset1.csv')
import pandas as pd
data = pd.read_csv(url)
st.write(data.head())


# Set up Streamlit app
st.set_page_config(page_title="AI-Integrated Dataset Dashboard", layout="wide")
st.title("AI-Integrated Interactive Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
sections = ["Home", "Exploratory Data Analysis", "AI Model", "Predictions"]
selection = st.sidebar.radio("Go to", sections)

if selection == "Home":
    st.header("Dataset Overview")
    st.write("### First 5 Rows of the Dataset")
    st.write(data.head())

    st.write("### Dataset Information")
    st.write(data.info())

    st.write("### Statistical Summary")
    st.write(data.describe(include='all'))

elif selection == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    # Distribution of categorical features
    st.write("### Categorical Feature Distributions")
    for col in ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'Placed']:
        st.write(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x=col, ax=ax)
        st.pyplot(fig)

    # Correlation heatmap
    st.write("### Correlation Matrix")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatter plot for numerical variables
    st.write("### Scatter Plot: SSC Percentage vs Salary")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='ssc_p', y='salary', hue='Placed', ax=ax)
    st.pyplot(fig)

elif selection == "AI Model":
    st.header("AI Model Training and Evaluation")

    # Preprocessing
    data['Placed'] = data['Placed'].map({'Yes': 1, 'No': 0})
    features = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    X = data[features]
    y = data['Placed']

    # Handle missing values if necessary
    if X.isnull().any().any() or y.isnull().any():
        st.warning("Warning: There are missing values in the dataset. Please handle them before training the model.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model Training
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d', ax=ax)
        st.pyplot(fig)

elif selection == "Predictions":
    st.header("Make Predictions")

    # User input form
    st.write("### Input Data for Prediction")
    input_data = {
        'ssc_p': st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0),
        'hsc_p': st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=75.0),
        'degree_p': st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=80.0),
        'etest_p': st.number_input("E-Test Percentage", min_value=0.0, max_value=100.0, value=60.0),
        'mba_p': st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=85.0),
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Placement Status"):
        prediction = model.predict(input_df)[0]
        status = "Placed" if prediction == 1 else "Not Placed"
       
