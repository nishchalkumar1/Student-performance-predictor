import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from src.predict import predict_result

def main():
    st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
    
    st.title("🎓 Student Performance Predictor")
    st.write("Predict whether a student will pass or fail based on their study hours.")
    
    # Sidebar for inputs
    st.sidebar.header("Input Features")
    hours = st.sidebar.slider("Study Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.pkl")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model not found! Please run the training pipeline first (python main.py).")
        return
        
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    # Make prediction
    prediction, probability = predict_result(model, hours)
    
    with col1:
        st.metric(label="Study Hours", value=f"{hours} hrs")
        
        if prediction == 1:
            st.success("Result: PASS 🎉")
        else:
            st.error("Result: FAIL 😢")
            
    with col2:
        if probability is not None:
            st.metric(label="Probability of Passing", value=f"{probability * 100:.2f}%")
            
            # Progress bar for visual probability
            st.progress(float(probability))
            
    # Display EDA and Visualizations
    st.divider()
    st.subheader("Model Visualizations")
    
    tabs = st.tabs(["Confusion Matrix", "Feature Importance", "Dataset View"])
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    with tabs[0]:
        st.write("Confusion matrix of the trained models:")
        cm_files = [f for f in os.listdir(models_dir) if f.startswith("cm_") and f.endswith(".png")]
        if cm_files:
            cols = st.columns(len(cm_files))
            for i, file in enumerate(cm_files):
                model_name = file.replace("cm_", "").replace(".png", "").replace("_", " ").title()
                cols[i].image(os.path.join(models_dir, file), caption=f"{model_name}")
        else:
            st.info("No confusion matrix plots found. Train the model first.")
            
    with tabs[1]:
        st.write("Feature Importance plots:")
        fi_files = [f for f in os.listdir(models_dir) if f.startswith("fi_") and f.endswith(".png")]
        if fi_files:
            cols = st.columns(len(fi_files))
            for i, file in enumerate(fi_files):
                model_name = file.replace("fi_", "").replace(".png", "").replace("_", " ").title()
                cols[i].image(os.path.join(models_dir, file), caption=f"{model_name}")
        else:
            st.info("No feature importance plots found.")
            
    with tabs[2]:
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "student_scores.csv")
            df = pd.read_csv(data_path)
            st.dataframe(df, use_container_width=True)
            st.line_chart(df.set_index("Hours")["Result"])
        except FileNotFoundError:
            st.info("Dataset not found.")

if __name__ == "__main__":
    main()
