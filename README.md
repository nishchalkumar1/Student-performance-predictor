# Student Performance Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)

A professional, portfolio-ready Machine Learning project that predicts whether a student will pass or fail based on their study hours. This project demonstrates an end-to-end ML pipeline, including Exploratory Data Analysis, feature engineering, model training (Logistic Regression, Decision Trees, Random Forests), evaluation, and deployment via a Streamlit web application.

---

## 🚀 Project Overview

The core objective of this project is to build an accurate predictive model that analyzes study hours and determines the likelihood of a student passing. 

The initial Jupyter Notebook implementation has been refactored into a scalable, production-quality project structure with modular scripts and an interactive web dashboard.

---

## 📂 Project Structure

```text
Student-Performance-Predictor/
├── data/
│   └── student_scores.csv      # Dataset
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
├── src/
│   ├── preprocess.py           # Data cleaning & train-test split
│   ├── train.py                # Model training (LR, DT, RF)
│   ├── evaluate.py             # Model evaluation & comparisons
│   ├── predict.py              # Prediction inference script
│   └── utils.py                # Helper functions (plotting, loading)
├── models/                     # Saved trained models and plot artifacts
├── app/
│   └── streamlit_app.py        # Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── main.py                     # Entry point for the ML pipeline
```

---

## 📊 Dataset Information

The dataset (`data/student_scores.csv`) is a simple structured dataset mapping study hours to an eventual Pass (1) or Fail (0) result.

- **Hours** (`float`): The number of hours a student studied.
- **Result** (`int`): The outcome. `0` for Fail, `1` for Pass.

---

## 🛠️ Technologies Used

- **Python** for core programming.
- **Pandas** & **NumPy** for data manipulation.
- **Matplotlib** & **Seaborn** for data visualization.
- **Scikit-Learn** for machine learning models and metrics.
- **Joblib** for saving and loading the trained model artifacts.
- **Streamlit** for building the interactive web application.

---

## ⚙️ Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Student-Performance-Predictor.git
   cd Student-Performance-Predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Instructions

### 1. Run the Machine Learning Pipeline

Execute the main pipeline to load data, train the models, generate evaluation artifacts, and save the best model.
```bash
python main.py
```

### 2. Launch the Streamlit Application

Start the interactive web application to input custom study hours and see live predictions.
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Comparison Results

The pipeline evaluates multiple algorithms to find the best fit.

| Model               | Evaluation                                                                 |
|---------------------|----------------------------------------------------------------------------|
| Logistic Regression | Evaluated using Accuracy, Classification Report, and Cross-Validation.     |
| Decision Tree       | Evaluated using Accuracy and Feature Importance.                           |
| Random Forest       | Evaluated using Accuracy, Feature Importance, and Cross-Validation.        |

*(Note: Exact accuracy scores depend on the train-test split and will be printed when running `main.py`)*

Visualizations including Confusion Matrices and Feature Importances are automatically saved into the `models/` directory during pipeline execution.

---

## 🔮 Future Improvements

- Incorporate more features into the dataset (e.g., previous grades, attendance, extracurriculars).
- Deploy the Streamlit app to Streamlit Community Cloud or Heroku.
- Add hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV` for all models.
- Implement more advanced algorithms (e.g., XGBoost, Gradient Boosting).
