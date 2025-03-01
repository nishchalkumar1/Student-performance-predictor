from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Trains a Decision Tree model with basic hyperparameter tuning."""
    param_grid = {
        'max_depth': [None, 3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_random_forest(X_train, y_train):
    """Trains a Random Forest model with basic hyperparameter tuning."""
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 3, 5],
        'min_samples_split': [2, 5]
    }
    # Using cv=2 since our dataset is quite small
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
