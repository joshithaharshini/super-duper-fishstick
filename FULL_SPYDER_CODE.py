import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Get the working directory of the current file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the backend files from the 'backend' folder
backend_path = os.path.join(working_dir, "backend")

# Load diabetes model
with open(os.path.join(backend_path, 'diabetes_model.sav'), 'rb') as file:
    diabetes_data = pickle.load(file)

diabetes_model = diabetes_data[0]

# Load metrics
with open(os.path.join(backend_path, 'metrics.pkl'), 'rb') as file:
    metrics = pickle.load(file)

# Load confusion matrix
conf_matrix = np.load(os.path.join(backend_path, 'confusion_matrix.npy'))

# Load feature importance
feature_importance = pd.read_csv(os.path.join(backend_path, 'feature_importance.csv'))

# Load model information
with open(os.path.join(backend_path, 'model_info.pkl'), 'rb') as file:
    model_info = pickle.load(file)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes risk Prediction', 'Heart Attack risk Prediction', 'Parkinson\'s Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'brain'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes risk Prediction':
    st.title('Diabetes Prediction using Machine Learning')
    st.subheader("Model Information")
    st.write(f"**Model Type**: {model_info['model_type']}")
    st.write(f"**Training Dataset**: {model_info['training_dataset']}")
    st.write(f"**Limitations**: {model_info['limitations']}")

    # Input data from user
    st.subheader("Enter Patient Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], key="gender")
    with col2:
        age = st.number_input('Age', min_value=0, max_value=120, step=1, key="age")
    with col3:
        hypertension = st.radio('Hypertension History', [0, 1], key="hypertension")
    with col1:
        heart_disease = st.radio('Heart Disease History', [0, 1], key="heart_disease")
    with col2:
        bmi = st.number_input('BMI Value', min_value=0.0, step=0.1, format="%.1f", key="bmi")
    with col3:
        HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, step=0.1, format="%.1f", key="HbA1c_level")
    with col1:
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, step=1, key="blood_glucose_level")

    # Prediction
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [
            1 if gender == 'Male' else 0,  # Encode gender
            age,
            hypertension,
            heart_disease,
            bmi,
            HbA1c_level,
            blood_glucose_level
        ]
        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)

    # Show additional metrics and visualizations
    st.subheader("Model Performance Metrics")
    st.write(f"**Accuracy**: {metrics['accuracy'] * 100:.2f}%")
    st.write(f"**Precision**: {metrics['precision'] * 100:.2f}%")
    st.write(f"**Recall**: {metrics['recall'] * 100:.2f}%")
    st.write(f"**F1-Score**: {metrics['f1_score'] * 100:.2f}%")

    st.subheader("Confusion Matrix")
    st.table(pd.DataFrame(conf_matrix, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))

    st.subheader("Feature Importance")
    st.bar_chart(feature_importance.set_index('Feature')['Importance'])
    
# Heart Disease Prediction Page
elif selected == 'Heart Attack risk Prediction':
    # Load model and scaler (assuming they are stored in the specified directory)
    model_path = r'D:\PREDICTIVE_APP\heart_attack_files\best_model.pkl'
    scaler_path = r'D:\PREDICTIVE_APP\heart_attack_files\scaler.pkl'

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the scaler for data transformation
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load the dataset for feature importance and column names
    heartattack_prediction_dataset_path = r'D:\PREDICTIVE_APP\heart_attack_files\heart.csv'
    heartattack_prediction_dataset = pd.read_csv(heartattack_prediction_dataset_path)

    X = heartattack_prediction_dataset.drop(columns=['output'])
    feature_names = X.columns

    # Page title
    st.title('Heart Attack Risk Predictor using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1, key="age_heart")
    with col2:
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female', key="sex")
    with col3:
        cp = st.selectbox('Chest Pain Types', [0, 1, 2, 3], key="cp")
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, step=1, key="trestbps")
    with col2:
        chol = st.number_input('Serum Cholesterol in mg/dl', min_value=100, max_value=600, step=1, key="chol")
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], key="fbs")
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], key="restecg")
    with col2:
        thalachh = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=220, step=1, key="thalachh")
    with col3:
        exng = st.selectbox('Exercise Induced Angina', [0, 1], key="exng")
    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, step=0.1, key="oldpeak")
    with col2:
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2], key="slope")
    with col3:
        caa = st.selectbox('Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4], key="caa")
    with col1:
        thall = st.selectbox('thal: 0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect', [0, 1, 2, 3], key="thall")

    # Initialize heart_prediction to None
    heart_prediction = None
    heart_diagnosis = ''
    health_tips = ""

    if st.button('Test Result'):
        # Prepare the user input list
        user_input = [
            age,         # age
            sex,         # sex (1 = male, 0 = female)
            cp,          # cp (chest pain types)
            trestbps,    # trestbps (resting blood pressure)
            chol,        # chol (serum cholesterol in mg/dl)
            fbs,         # fbs (fasting blood sugar > 120 mg/dl)
            restecg,     # restecg (resting electrocardiographic results)
            thalachh,    # thalachh (maximum heart rate achieved)
            exng,        # exng (exercise induced angina)
            oldpeak,     # oldpeak (ST depression induced by exercise)
            slope,       # slope (slope of the peak exercise ST segment)
            caa,          # ca (major vessels colored by fluoroscopy)
            thall        # thall (thal: 0 = normal, 1 = fixed defect, 2 = reversible defect)
        ]
        
        # Ensure all inputs are float for model prediction
        user_input = [float(x) for x in user_input]

        # Transform the input using the saved scaler
        user_input_scaled = scaler.transform([user_input])

        # Make prediction
        heart_prediction = model.predict(user_input_scaled)

        # Now you can safely reference heart_prediction[0]
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is at risk of heart attack'
            health_tips = """
            **Health Tips:**
            - It's important to consult a healthcare professional for further diagnosis and treatment.
            - Consider making lifestyle changes, such as reducing salt intake and maintaining a healthy weight.
            - Regular exercise is crucial, but always seek medical advice before starting an exercise program.
            - Medications like statins may be prescribed to manage cholesterol levels.
            - Manage stress and avoid smoking.
            """
        else:
            heart_diagnosis = 'The person is not at risk of heart attack'
            health_tips = """
            **Health Tips:**
            - Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.
            - Focus on cardiovascular health by engaging in activities like walking, swimming, or cycling.
            - Stay hydrated and avoid excessive alcohol consumption.
            - Regularly check your blood pressure, cholesterol, and blood sugar levels.
            - Keep a healthy weight and avoid high-fat foods.
            """

        st.success(heart_diagnosis)

        # Feature Importance with Plotly
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

            fig = px.bar(feature_importance_df, x='Feature', y='Importance',
                         title="Feature Importance",
                         labels={'Feature': 'Features', 'Importance': 'Importance Score'})
            st.plotly_chart(fig)
        else:
            st.warning("Model does not have feature importance attribute.")

    # Display health tips after prediction
    if heart_prediction is not None:
        st.write(health_tips)

# Parkinson's Disease Prediction Page
elif selected == "Parkinson's Disease Prediction":
    # Load model and scaler
    park_model_path = os.path.join(r'D:\PREDICTIVE_APP\parkinsons_files\park_best_model.pkl')
    park_scaler_path = os.path.join(r'D:\PREDICTIVE_APP\parkinsons_files\park_scaler.pkl')

    with open(park_model_path, 'rb') as model_file:
        park_model = pickle.load(model_file)

    with open(park_scaler_path, 'rb') as scaler_file:
        park_scaler = pickle.load(scaler_file)

    # Input for Parkinson's disease prediction
    st.title("Parkinson's Disease Prediction")

    st.subheader("Enter Patient's Voice Features")

    # List of 9 features after dropping correlated ones (adjusted from APPLE's output)
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
        "RPDE", "DFA", "spread1", "spread2", "D2"
    ]

    # Ensure the number of features is exactly as expected by the model
    user_input = []
    for feature in features:
        value = st.number_input(f"{feature}", key=feature)
        user_input.append(value)

    if st.button("Predict Parkinson's Disease"):
        # Convert input to the correct shape and scale the data
        user_input = np.array(user_input).reshape(1, -1)  # Reshape for a single prediction

        # Check if the number of features matches the expected count
        expected_feature_count = len(park_scaler.mean_)  # Using mean_ attribute of StandardScaler to check feature count

        # Compare with 9 features from the backend model
        if user_input.shape[1] == 9:  # We are expecting 9 features as per the model training
            user_input_scaled = park_scaler.transform(user_input)

            # Make prediction
            park_prediction = park_model.predict(user_input_scaled)

            if park_prediction[0] == 1:
                st.success("The person is predicted to have Parkinson's disease")
            else:
                st.success("The person is predicted to be free of Parkinson's disease")
        else:
            st.error(f"Input features mismatch. Expected 9 features, but received {user_input.shape[1]} features.")
    
# Footer
st.markdown(
    """
    <hr>
    <footer>
        <p style="text-align: center;">
            Health Assistant © 2025. All Rights Reserved.
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)