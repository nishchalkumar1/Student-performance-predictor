import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    """Cleans the dataset by removing nulls or duplicates if any."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def feature_engineering(df):
    """Placeholder for feature engineering, e.g., creating polynomial features if needed."""
    # Since we only have 'Hours', we will just return it as is for now.
    # In a more complex scenario, we might add Hours^2 etc.
    return df

def prepare_data(df, target_col='Result', test_size=0.2, random_state=42):
    """Splits the data into features and target, then into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
