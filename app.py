import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("c:\\Users\\Admin\\Saved Games\\DJS_Compute_tasks\\log_reg_model.pkl")
scaler = joblib.load("c:\\Users\\Admin\\Saved Games\\DJS_Compute_tasks\\scaler.pkl")
model_columns = joblib.load("c:\\Users\\Admin\\Saved Games\\DJS_Compute_tasks\\columns.pkl")

st.title('Asthma Prediction App ')
st.markdown("""This app predicts whether a person has asthma based on their health profile.""")

st.header('Enter Patient Details:')

age = st.slider('Age', 0, 100, 30)
gender = st.selectbox('Gender', ('Female', 'Male', 'Other'))
bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ('Current', 'Former', 'Never'))
family_history = st.selectbox('Family History of Asthma', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
allergies = st.selectbox('Allergies', ('None', 'Dust', 'Multiple', 'Pollen', 'Pets'))
air_pollution_level = st.selectbox('Air Pollution Level', ('Low', 'Moderate', 'High'))
physical_activity_level = st.selectbox('Physical Activity Level', ('Sedentary', 'Moderate', 'Active'))
occupation_type = st.selectbox('Occupation Type', ('Indoor', 'Outdoor'))
comorbidities = st.selectbox('Comorbidities', ('None', 'Both', 'Diabetes', 'Hypertension'))
medication_adherence = st.slider('Medication Adherence (0.0 - 1.0)', 0.0, 1.0, 0.5)
number_of_er_visits = st.number_input('Number of ER Visits', min_value=0, max_value=10, value=0)
peak_expiratory_flow = st.number_input('Peak Expiratory Flow (PEF)', min_value=100.0, max_value=800.0, value=450.0)
feno_level = st.number_input('FeNO Level', min_value=10.0, max_value=100.0, value=30.0)

if st.button('Predict'):
    user_data = {
        'Age': age,
        'BMI': bmi,
        'Family_History': family_history,
        'Medication_Adherence': medication_adherence,
        'Number_of_ER_Visits': number_of_er_visits,
        'Peak_Expiratory_Flow': peak_expiratory_flow,
        'FeNO_Level': feno_level,
        'Has_Asthma': 0  # Placeholder- not used in prediction
    }

    # One-hot encoding manually for categorical features
    # Gender
    user_data['Gender_Male'] = 1 if gender == 'Male' else 0
    user_data['Gender_Other'] = 1 if gender == 'Other' else 0

    # Smoking_Status
    user_data['Smoking_Status_Former'] = 1 if smoking_status == 'Former' else 0
    user_data['Smoking_Status_Never'] = 1 if smoking_status == 'Never' else 0

    # Allergies
    user_data['Allergies_Multiple'] = 1 if allergies == 'Multiple' else 0
    user_data['Allergies_Pets'] = 1 if allergies == 'Pets' else 0
    user_data['Allergies_Pollen'] = 1 if allergies == 'Pollen' else 0

    # Air_Pollution_Level
    user_data['Air_Pollution_Level_Low'] = 1 if air_pollution_level == 'Low' else 0
    user_data['Air_Pollution_Level_Moderate'] = 1 if air_pollution_level == 'Moderate' else 0

    # Physical_Activity_Level
    user_data['Physical_Activity_Level_Moderate'] = 1 if physical_activity_level == 'Moderate' else 0
    user_data['Physical_Activity_Level_Sedentary'] = 1 if physical_activity_level == 'Sedentary' else 0

    # Occupation_Type
    user_data['Occupation_Type_Outdoor'] = 1 if occupation_type == 'Outdoor' else 0

    # Comorbidities
    user_data['Comorbidities_Diabetes'] = 1 if comorbidities == 'Diabetes' else 0
    user_data['Comorbidities_Hypertension'] = 1 if comorbidities == 'Hypertension' else 0

    # BMI_Category
    user_data['BMI_Category_Obese'] = 1 if bmi >= 30 else 0
    user_data['BMI_Category_Overweight'] = 1 if 25 <= bmi < 30 else 0
    user_data['BMI_Category_Underweight'] = 1 if bmi < 18.5 else 0

    # Create DataFrame and ensure all columns match model_columns
    input_df = pd.DataFrame([user_data])
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    # Scale numeric features
    input_df_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_df_scaled)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Based on the input features, the model predicts the person **has asthma**.')
    else:
        st.success('Based on the input features, the model predicts the person **does not have asthma**.')