"""
Heart Disease Detection Streamlit Application
=============================================

This application loads a pre-trained heart disease detection model (XGBoost/sklearn)
and provides an interactive UI for users to input patient medical data and receive
risk predictions.

Features:
- 20 input fields for patient medical data
- Real-time prediction using the trained model
- Risk assessment with color-coded output
- Input validation and preprocessing
- Session state management for better UX

Author: ML Team
Date: 2024
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
        font-size: 18px;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
        font-size: 18px;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== FEATURE CONFIGURATION ====================
# Define all 20 features in the exact order they were used in model training
FEATURES = [
    'Age',
    'Gender',
    'Blood Pressure',
    'Cholesterol Level',
    'Exercise Habits',
    'Smoking',
    'Family Heart Disease',
    'Diabetes',
    'BMI',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
    'Alcohol Consumption',
    'Stress Level',
    'Sleep Hours',
    'Sugar Consumption',
    'Triglyceride Level',
    'Fasting Blood Sugar',
    'CRP Level',
    'Homocysteine Level'
]

# Feature descriptions for user guidance
FEATURE_DESCRIPTIONS = {
    'Age': 'Patient age in years (typically 18-100)',
    'Gender': 'Gender (0: Female, 1: Male)',
    'Blood Pressure': 'Systolic blood pressure in mmHg',
    'Cholesterol Level': 'Total cholesterol in mg/dL',
    'Exercise Habits': 'Exercise frequency per week (0-7)',
    'Smoking': 'Smoking status (0: No, 1: Yes)',
    'Family Heart Disease': 'Family history of heart disease (0: No, 1: Yes)',
    'Diabetes': 'Diabetes status (0: No, 1: Yes)',
    'BMI': 'Body Mass Index',
    'High Blood Pressure': 'High blood pressure status (0: No, 1: Yes)',
    'Low HDL Cholesterol': 'Low HDL cholesterol status (0: No, 1: Yes)',
    'High LDL Cholesterol': 'High LDL cholesterol status (0: No, 1: Yes)',
    'Alcohol Consumption': 'Alcohol consumption (drinks per week, 0-20)',
    'Stress Level': 'Self-reported stress level (1-10 scale)',
    'Sleep Hours': 'Average sleep hours per night (0-12)',
    'Sugar Consumption': 'Sugar consumption (grams per day)',
    'Triglyceride Level': 'Triglyceride level in mg/dL',
    'Fasting Blood Sugar': 'Fasting blood sugar in mg/dL',
    'CRP Level': 'C-reactive protein level in mg/L',
    'Homocysteine Level': 'Homocysteine level in µmol/L'
}

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model(model_path='heart_model.pkl'):
    """
    Load the pre-trained heart disease model using joblib.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at '{model_path}'")
        st.info("Please ensure 'heart_model.pkl' is in the same directory as this script.")
        return None


# ==================== PREDICTION FUNCTION ====================
def get_prediction(model, features_input):
    """
    Generate prediction from the model using patient input.
    
    Args:
        model: Trained model object
        features_input (list): List of 20 feature values in correct order
        
    Returns:
        tuple: (prediction, prediction_probability)
            - prediction: 0 (No risk) or 1 (At risk)
            - probability: Confidence score (0-1)
    """
    try:
        # Convert to numpy array with shape (1, 20)
        features_array = np.array(features_input).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_array)[0]
            risk_probability = probability[1]  # Probability of disease
        else:
            # For models without predict_proba, estimate based on decision function
            if hasattr(model, 'decision_function'):
                score = model.decision_function(features_array)[0]
                risk_probability = 1 / (1 + np.exp(-score))  # Sigmoid conversion
            else:
                risk_probability = float(prediction)
        
        return int(prediction), round(risk_probability, 4)
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None


# ==================== UI COMPONENTS ====================
def display_header():
    """Display the application header and description."""
    st.title("❤️ Heart Disease Detection System")
    st.markdown("""
    This application uses machine learning to assess the risk of heart disease 
    based on patient medical data. Please enter accurate medical information below.
    """)
    st.divider()


def display_input_form():
    """
    Create and manage input form for all 20 features.
    
    Returns:
        list: Array of 20 feature values in correct order
    """
    st.subheader("📋 Patient Medical Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    input_values = []
    
    # Organize features into groups for better UX
    feature_groups = {
        "Demographics": FEATURES[:2],
        "Vital Signs": FEATURES[2:4],
        "Lifestyle": FEATURES[4:6],
        "Medical History": FEATURES[6:9],
        "Cardiac Risk Markers": FEATURES[9:13],
        "Lifestyle & Stress": FEATURES[13:15],
        "Dietary & Blood Markers": FEATURES[15:20]
    }
    
    # Render input fields grouped by category
    for group_name, group_features in feature_groups.items():
        st.markdown(f"**{group_name}**")
        
        # Alternate between columns
        cols = st.columns(2)
        
        for idx, feature in enumerate(group_features):
            col = cols[idx % 2]
            
            with col:
                # Determine input type based on feature
                if feature == 'Gender':
                    # Gender field - special handling
                    value = st.selectbox(
                        label=feature,
                        options=[0, 1],
                        format_func=lambda x: "Male" if x == 1 else "Female",
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                elif feature in ['Smoking', 'Family Heart Disease', 
                               'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol',
                               'High LDL Cholesterol']:
                    # Binary features (Yes/No)
                    value = st.selectbox(
                        label=feature,
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                elif feature == 'Exercise Habits':
                    # Exercise frequency per week
                    exercise_options = {
                        0: "Never (0 days/week)",
                        1: "Rarely (1 day/week)",
                        2: "Occasionally (2 days/week)",
                        3: "Moderate (3 days/week)",
                        4: "Regular (4 days/week)",
                        5: "Very Regular (5 days/week)",
                        6: "Almost Daily (6 days/week)",
                        7: "Daily (7 days/week)"
                    }
                    value = st.selectbox(
                        label=feature,
                        options=list(exercise_options.keys()),
                        format_func=lambda x: exercise_options[x],
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                elif feature == 'Alcohol Consumption':
                    # Alcohol consumption per week
                    alcohol_options = {
                        0: "None (0 drinks/week)",
                        1: "Light (1 drink/week)",
                        2: "Light (2 drinks/week)",
                        3: "Light (3 drinks/week)",
                        4: "Moderate (4 drinks/week)",
                        5: "Moderate (5 drinks/week)",
                        6: "Moderate (6 drinks/week)",
                        7: "Heavy (7 drinks/week)",
                        8: "Heavy (8 drinks/week)",
                        9: "Heavy (9 drinks/week)",
                        10: "Heavy (10+ drinks/week)",
                        11: "Very Heavy (11+ drinks/week)",
                        12: "Very Heavy (12+ drinks/week)",
                        13: "Very Heavy (13+ drinks/week)",
                        14: "Very Heavy (14+ drinks/week)",
                        15: "Very Heavy (15+ drinks/week)",
                        16: "Very Heavy (16+ drinks/week)",
                        17: "Very Heavy (17+ drinks/week)",
                        18: "Very Heavy (18+ drinks/week)",
                        19: "Very Heavy (19+ drinks/week)",
                        20: "Very Heavy (20 drinks/week)"
                    }
                    value = st.selectbox(
                        label=feature,
                        options=list(alcohol_options.keys()),
                        format_func=lambda x: alcohol_options[x],
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                elif feature == 'Sleep Hours':
                    # Sleep hours per night
                    sleep_options = {
                        0: "None (0 hours)",
                        1: "Very Poor (1 hour)",
                        2: "Very Poor (2 hours)",
                        3: "Poor (3 hours)",
                        4: "Poor (4 hours)",
                        5: "Below Average (5 hours)",
                        6: "Below Average (6 hours)",
                        7: "Adequate (7 hours)",
                        8: "Good (8 hours)",
                        9: "Good (9 hours)",
                        10: "Excellent (10 hours)",
                        11: "Excellent (11 hours)",
                        12: "Excessive (12 hours)"
                    }
                    value = st.selectbox(
                        label=feature,
                        options=list(sleep_options.keys()),
                        format_func=lambda x: sleep_options[x],
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                elif feature == 'Sugar Consumption':
                    # Sugar consumption per day in grams
                    sugar_options = {
                        0: "None (0g/day)",
                        25: "Low (25g/day)",
                        50: "Low-Moderate (50g/day)",
                        75: "Moderate (75g/day)",
                        100: "Moderate (100g/day)",
                        125: "Moderate-High (125g/day)",
                        150: "High (150g/day)",
                        175: "High (175g/day)",
                        200: "Very High (200g/day)",
                        250: "Very High (250g/day)",
                        300: "Excessive (300g/day)",
                        350: "Excessive (350g/day)",
                        400: "Excessive (400g/day)",
                        450: "Excessive (450g/day)",
                        500: "Excessive (500g/day)"
                    }
                    value = st.selectbox(
                        label=feature,
                        options=list(sugar_options.keys()),
                        format_func=lambda x: sugar_options[x],
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                elif feature == 'Stress Level':
                    # Stress level on a scale
                    value = st.slider(
                        label=feature,
                        min_value=1,
                        max_value=10,
                        step=1,
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                else:
                    # Numeric input with appropriate ranges
                    value = st.number_input(
                        label=feature,
                        value=0.0,
                        step=0.1,
                        help=FEATURE_DESCRIPTIONS[feature],
                        key=f"input_{feature}"
                    )
                
                input_values.append(value)
    
    return input_values


def display_prediction_result(prediction, probability):
    """
    Display prediction results with risk assessment.
    
    Args:
        prediction (int): 0 or 1 (no risk or at risk)
        probability (float): Confidence score 0-1
    """
    st.divider()
    st.subheader("🔬 Prediction Result")
    
    # Determine risk level and styling
    if prediction == 1:
        if probability >= 0.8:
            risk_level = "High Risk"
            risk_class = "risk-high"
            risk_color = "#d32f2f"
            risk_icon = "🚨"
        else:
            risk_level = "Medium Risk"
            risk_class = "risk-medium"
            risk_color = "#f57c00"
            risk_icon = "⚠️"
    else:
        risk_level = "Low Risk"
        risk_class = "risk-low"
        risk_color = "#388e3c"
        risk_icon = "✅"
    
    # Display results with colored boxes for better visibility
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Status with colored background
        st.markdown(f"""
        <div style="background-color: {risk_color}; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: 500;">Risk Status</p>
            <h2 style="color: white; margin: 10px 0 0 0; font-size: 28px;">{risk_icon} {risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Risk Probability with colored background
        st.markdown(f"""
        <div style="background-color: {risk_color}; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <p style="color: white; margin: 0; font-size: 14px; font-weight: 500;">Risk Probability</p>
            <h2 style="color: white; margin: 10px 0 0 0; font-size: 28px;">{probability * 100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed recommendation
    st.markdown("---")
    
    if prediction == 1:
        st.warning(
            f"⚠️ **{risk_level}**: This patient shows indicators of potential heart disease risk. "
            f"We recommend:\n"
            f"- Consultation with a cardiologist\n"
            f"- Further diagnostic testing (ECG, stress test)\n"
            f"- Regular monitoring of vital signs\n"
            f"- Lifestyle modifications as recommended by healthcare professionals"
        )
    else:
        st.success(
            "✅ **Low Risk**: This patient shows minimal indicators of heart disease. "
            "However, continue regular health checkups and maintain a healthy lifestyle."
        )


def display_sidebar_info():
    """Display information and instructions in the sidebar."""
    with st.sidebar:
        st.markdown("### 📖 How to Use")
        st.info(
            "1. Enter all patient medical information\n"
            "2. Click 'Get Prediction' button\n"
            "3. Review the risk assessment results\n"
            "4. Consult healthcare professionals for medical decisions"
        )
        
        st.markdown("### ⚕️ Important Notice")
        st.warning(
            "This tool is for **informational purposes only** and should not replace "
            "professional medical advice. Always consult with qualified healthcare "
            "professionals for diagnosis and treatment decisions."
        )
        
        st.markdown("### 📊 Model Information")
        st.info(
            "- **Model Type**: XGBoost Classification\n"
            "- **Features**: 20 medical indicators\n"
            "- **Output**: Binary classification (At Risk / Not At Risk)\n"
            "- **Probability**: Confidence score (0-100%)"
        )
        
        st.markdown("### 🔧 Technical Details")
        with st.expander("View feature list"):
            features_df = pd.DataFrame({
                'Feature': FEATURES,
                'Description': [FEATURE_DESCRIPTIONS[f] for f in FEATURES]
            })
            st.dataframe(features_df, use_container_width=True, hide_index=True)


# ==================== MAIN APPLICATION ====================
def main():
    """Main application flow."""
    
    # Display header
    display_header()
    
    # Display sidebar info
    display_sidebar_info()
    
    # Load the model
    model = load_model('heart_model.pkl')
    
    if model is None:
        st.stop()
    
    # Create input form
    st.markdown("### Please provide the following patient information:")
    input_values = display_input_form()
    
    # Prediction button
    col1, col2 = st.columns([1, 4])
    
    with col1:
        predict_button = st.button(
            "🔬 Get Prediction",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.markdown(
            "<p style='text-align: center; color: gray; font-size: 14px;'>"
            "Click to analyze patient data</p>",
            unsafe_allow_html=True
        )
    
    # Generate prediction
    if predict_button:
        # Validate inputs
        if len(input_values) != 20:
            st.error("❌ Error: Not all fields were filled. Please complete the form.")
            st.stop()
        
        # Convert to numpy array for prediction
        try:
            prediction, probability = get_prediction(model, input_values)
            
            if prediction is not None:
                # Display results
                display_prediction_result(prediction, probability)
                
                # Store prediction in session state for reference
                st.session_state.last_prediction = {
                    'prediction': prediction,
                    'probability': probability,
                    'features': input_values,
                    'timestamp': pd.Timestamp.now()
                }
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Display previous prediction if available (session state)
    if 'last_prediction' in st.session_state:
        with st.expander("📝 Previous Prediction Summary"):
            prev = st.session_state.last_prediction
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Timestamp**: {prev['timestamp']}")
                st.write(f"**Risk Probability**: {prev['probability'] * 100:.2f}%")
            with col2:
                st.write(f"**Status**: {'At Risk' if prev['prediction'] == 1 else 'Not At Risk'}")


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    main()


