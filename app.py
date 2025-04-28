import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ====== Load the DataFrame ======
df = pd.read_pickle('df.pkl')  

# ====== Prepare Data ======
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Preprocessing Pipeline ======
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# ====== Train Models ======
Final_models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=105),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=150)
}

trained_models = {}
for name, model in Final_models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline

models = trained_models

# ====== Streamlit Frontend ======

# Title
def display_title():
    st.markdown("<h1 style='text-align: center; color: crimson;'>❤️ Heart Disease Predictor</h1>", unsafe_allow_html=True)
    st.markdown("""
        Welcome to the Heart Disease Prediction App!  
        Provide your information and get predictions from two advanced machine learning models.
    """)

# Collect User Input
def get_user_input():
    st.markdown("## 📝 Enter Your Details:")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=30)
        sex = st.selectbox('Sex', df['Sex'].unique())
        resting_bp = st.number_input('RestingBP', min_value=0, value=120)
        cholesterol = st.number_input('Cholesterol', min_value=0, value=200)
        max_hr = st.number_input('MaxHR', min_value=0, value=150)

    with col2:
        chest_pain_type = st.selectbox('ChestPainType', df['ChestPainType'].unique())
        fasting_bs = st.selectbox('FastingBS', df['FastingBS'].unique())
        resting_ecg = st.selectbox('RestingECG', df['RestingECG'].unique())
        exercise_angina = st.selectbox('ExerciseAngina', df['ExerciseAngina'].unique())
        st_slope = st.selectbox('ST_Slope', df['ST_Slope'].unique())

    oldpeak = st.slider('Oldpeak', float(df['Oldpeak'].min()), float(df['Oldpeak'].max()), 1.0)

    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    
    return input_data

# Prediction Function
def predict_heart_disease(input_data):
    input_df = pd.DataFrame([input_data])
    results = {}

    for model_name, model_pipeline in models.items():
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]
        results[model_name] = {'prediction': prediction, 'probability': probability}

    return results

# Main App
def main():
    display_title()
    user_input = get_user_input()

    if st.button('🔍 Predict Heart Disease'):
        predictions = predict_heart_disease(user_input)

        st.markdown("## 🧠 Prediction Results:")

        col1, col2 = st.columns(2)
        columns = [col1, col2]

        for idx, (model_name, result) in enumerate(predictions.items()):
            pred = result['prediction']
            prob = result['probability']

            with columns[idx]:
                st.subheader(f"{model_name}")
                if pred == 1:
                    st.error(f"Result: Positive for Heart Disease\nConfidence: {prob*100:.2f}%")
                else:
                    st.success(f"Result: No Heart Disease\nConfidence: {100 - prob*100:.2f}%")

        st.markdown("---")
        if st.button('🔄 Try Again'):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
