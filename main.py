import os
import joblib
from src.utils import load_data, plot_confusion_matrix, plot_feature_importance
from src.preprocess import clean_data, prepare_data
from src.train import train_logistic_regression, train_decision_tree, train_random_forest
from src.evaluate import evaluate_model, compare_models, perform_cross_validation
import matplotlib.pyplot as plt

def main():
    print("--- Student Performance Predictor Pipeline ---")
    
    # 1. Load Data
    data_path = "data/student_scores.csv"
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    # 2. Preprocess Data
    print("Cleaning and preparing data...")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Full X, y for cross validation
    X = df.drop(columns=['Result'])
    y = df['Result']
    
    # 3. Train Models
    print("Training models...")
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train)
    }
    
    # 4. Evaluate Models
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        acc, report, y_pred = evaluate_model(model, X_test, y_test)
        cv_score = perform_cross_validation(model, X, y)
        print(f"\n{name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Cross-Validation Accuracy: {cv_score:.4f}")
        print("Classification Report:")
        print(report)
        
        # Save confusion matrix plot
        os.makedirs("models", exist_ok=True)
        cm_fig = plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix: {name}")
        cm_fig.savefig(f"models/cm_{name.replace(' ', '_').lower()}.png")
        plt.close(cm_fig)
        
        # Save feature importance if applicable
        if name in ["Decision Tree", "Random Forest"]:
            fi_fig = plot_feature_importance(model, ['Hours'], title=f"Feature Importance: {name}")
            if fi_fig:
                fi_fig.savefig(f"models/fi_{name.replace(' ', '_').lower()}.png")
                plt.close(fi_fig)
                
    # 5. Compare Models
    print("\n--- Model Comparison ---")
    comparison_df = compare_models(models, X_test, y_test)
    print(comparison_df.to_string(index=False))
    
    # 6. Save Best Model
    # Simple selection: pick model with highest accuracy on test set
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]
    print(f"\nSaving the best model ({best_model_name}) to models/best_model.pkl...")
    joblib.dump(best_model, "models/best_model.pkl")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
