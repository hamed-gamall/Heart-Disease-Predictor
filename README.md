# ❤️ Heart Disease Predictor

A powerful machine learning project that analyzes patient medical data to predict heart disease risk.  
It identifies key patterns like age, chest pain type, blood pressure, cholesterol, and heart rate to detect high-risk individuals.  
This project provides early insights for patients and supports data-driven medical decision making.

---

## 📸 Project Preview
![Heart Disease](./image1.png)

---

## 📂 Dataset Features
- **Age**: Age of the patient  
- **Sex**: M = Male, F = Female  
- **ChestPainType**: TA, ATA, NAP, ASY  
- **RestingBP**: Resting blood pressure  
- **Cholesterol**: Serum cholesterol  
- **FastingBS**: Fasting blood sugar (1 if >120 mg/dl)  
- **RestingECG**: ECG result (Normal, ST, LVH)  
- **MaxHR**: Maximum heart rate achieved  
- **ExerciseAngina**: Exercise-induced angina (Y/N)  
- **Oldpeak**: ST depression  
- **ST_Slope**: Up, Flat, Down  
- **HeartDisease**: Target (1 = Heart disease, 0 = Normal)

---

## 🔍 Key Analysis Insights
- Most heart disease cases occur after **age 50**.  
- **ASY chest pain** is the most common among heart disease patients.  
- Around **90% of heart disease cases are male**.  
- Patients with heart disease often show:
  - Lower **MaxHR**
  - Higher **RestingBP**
  - Higher **Cholesterol**
  - Mostly **Flat ST Slope**
  - More **ST-type ECG** abnormalities  
- MaxHR decreases as age increases.  
- Age, RestingBP, and HeartDisease show a **moderate correlation (~30%)**.

---

## 🧠 Workflow
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Visualization  
- Model Training  
- Evaluation (Accuracy, Precision, Recall, F1-score)

---
### 🌐 6. Deployment
- Interactive dashboards using:
  - ✅ Streamlit [👉 Try App NOW](https://heart-disease-predictio.streamlit.app/)
 

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook

