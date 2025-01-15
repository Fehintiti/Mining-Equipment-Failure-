#!/usr/bin/env python
# coding: utf-8

# In[15]:


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load saved models ,scaler and label_encoder
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Title and Introduction
st.title("Case Study: Predicting Equipment Failure in Mining Operations")
st.markdown("""
This app predicts whether a piece of mining equipment is likely to fail based on operational data.
You can input the machine's parameters on the left (click on the arrow), and the app will provide a prediction.
""")

# Sidebar for User Inputs
st.sidebar.header("User Input Features")

def user_input_features():
    air_temp = st.sidebar.slider('Air Temperature [K]', 290, 320, 300)
    process_temp = st.sidebar.slider('Process Temperature [K]', 300, 350, 320)
    rotational_speed = st.sidebar.slider('Rotational Speed [rpm]', 1000, 3000, 1500)
    torque = st.sidebar.slider('Torque [Nm]', 10, 80, 50)
    tool_wear = st.sidebar.slider('Tool Wear [min]', 0, 250, 125)
    machine_type = st.sidebar.selectbox('Type', ['L', 'M', 'H'])

    data = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type': machine_type
    }
    return pd.DataFrame(data, index=[0])

# Collect user input
input_df = user_input_features()

# Display user inputs
st.subheader("User Input Features")
st.write(input_df)

# Preprocess Input Data
# Encode the 'Type' column
input_df['Type'] = label_encoder.transform(input_df['Type'])

# Select numerical columns for scaling
numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Scale the numerical data
scaled_numerical = scaler.transform(input_df[numerical_columns])

# Combine scaled numerical features with the encoded 'Type' column
processed_input = pd.DataFrame(scaled_numerical, columns=numerical_columns)
processed_input['Type'] = input_df['Type'].values  # Add the encoded 'Type' column

# Reorder columns to match the training data's order
processed_input = processed_input[['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]

# Rename columns to match training data feature names
processed_input = processed_input.rename(columns={
    'Air temperature [K]': 'Air temperature K',
    'Process temperature [K]': 'Process temperature K',
    'Rotational speed [rpm]': 'Rotational speed rpm',
    'Torque [Nm]': 'Torque Nm',
    'Tool wear [min]': 'Tool wear min'
})

# Make Predictions
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

# Display Predictions
st.subheader("Prediction")
st.write("**Failure**" if prediction[0] == 1 else "**No Failure**")

st.subheader("Prediction Probability")
st.write(f"**Failure:** {prediction_proba[0][1]:.2f}, **No Failure:** {prediction_proba[0][0]:.2f}")

# Optional Feature Importance Plot
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importances")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    st.pyplot(plt)

if st.checkbox("Show Feature Importance"):
    st.subheader("Feature Importance")
    st.write("Visualizing the importance of each feature in predicting failures:")
    plot_feature_importance(model, list(processed_input.columns))


# In[ ]:




