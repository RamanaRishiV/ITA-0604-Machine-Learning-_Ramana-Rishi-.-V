
import os
import urllib.request
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "attrisense_model.pkl")

# IBM HR Analytics dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/"
    "main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Features used for prediction
FEATURES = [
    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "YearsAtCompany",
    "WorkLifeBalance",
    "EnvironmentSatisfaction"
]

TARGET = "Attrition"


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Downloading IBM HR Analytics dataset...")

        urllib.request.urlretrieve(DATASET_URL, DATA_PATH)

        print("Dataset downloaded successfully.")

    return DATA_PATH


def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    download_dataset()
    df = pd.read_csv(DATA_PATH)

    # Separate input and target
    X = df[FEATURES].copy()
    y = df[TARGET].map({"Yes": 1, "No": 0})

    # Identify column types
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # Preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Random Forest model
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Complete machine learning pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train and evaluate
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nATTRISENSE MODEL TRAINED SUCCESSFULLY")
    print("-------------------------------------")
    print(f"Dataset records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Likely to Stay", "Likely to Leave"],
        zero_division=0
    ))

    # Retrain on the complete dataset
    model.fit(X, y)

    # Save trained model and accuracy
    joblib.dump({
        "model": model,
        "accuracy": accuracy,
        "features": FEATURES
    }, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")

    return model, accuracy


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()

    saved_data = joblib.load(MODEL_PATH)

    return saved_data["model"], saved_data["accuracy"]


def predict_employee(employee_data):
    model, accuracy = load_model()

    employee_df = pd.DataFrame(
        [employee_data],
        columns=FEATURES
    )

    prediction = int(model.predict(employee_df)[0])

    probabilities = model.predict_proba(employee_df)[0]

    risk_probability = float(probabilities[1]) * 100

    result = (
        "Likely to Leave"
        if prediction == 1
        else "Likely to Stay"
    )

    return {
        "prediction": result,
        "attrition_probability": round(risk_probability, 2),
        "model_accuracy": round(accuracy * 100, 2)
    }


if __name__ == "__main__":
    train_model()
