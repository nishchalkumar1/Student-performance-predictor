import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report, y_pred

def compare_models(models_dict, X_test, y_test):
    """Compares multiple models and returns a DataFrame of their accuracies."""
    results = []
    for name, model in models_dict.items():
        acc, _, _ = evaluate_model(model, X_test, y_test)
        results.append({'Model': name, 'Accuracy': acc})
    
    df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    return df_results

def perform_cross_validation(model, X, y, cv=3):
    """Performs cross-validation and returns the mean accuracy."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()
