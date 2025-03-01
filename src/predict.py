import joblib
import os
import pandas as pd

def load_model(model_path="models/best_model.pkl"):
    """Loads the trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def predict_result(model, hours):
    """Predicts pass/fail based on study hours."""
    # The model expects a 2D array-like input, e.g., DataFrame with 'Hours' column
    input_data = pd.DataFrame({'Hours': [hours]})
    prediction = model.predict(input_data)[0]
    
    # Try to get probability if the model supports it
    try:
        probability = model.predict_proba(input_data)[0][1] # Probability of passing (class 1)
    except AttributeError:
        probability = None
        
    return prediction, probability
