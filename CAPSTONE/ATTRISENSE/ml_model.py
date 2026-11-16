
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# ATTRISENSE - EMPLOYEE ATTRITION PREDICTION SYSTEM
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(
    DATA_DIR,
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "attrisense_model.pkl"
)

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "IamVasil/IBM-HR-Analytics-Employee-Attrition-Performance/"
    "main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

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

# Risk categories (initial thresholds)
LOW_RISK_LIMIT = 30
MEDIUM_RISK_LIMIT = 60


# =====================================================
# DATASET MANAGEMENT
# =====================================================

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Downloading IBM HR Analytics dataset...")

        urllib.request.urlretrieve(
            DATASET_URL,
            DATA_PATH
        )

        print("Dataset downloaded successfully.")

    return DATA_PATH


# =====================================================
# MODEL TRAINING
# =====================================================

def train_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    download_dataset()

    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES].copy()
    y = df[TARGET].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Unexpected values found in Attrition column.")

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    # Random Forest classifier
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

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

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\n======================================")
    print(" ATTRISENSE MODEL TRAINED SUCCESSFULLY")
    print("======================================")

    print(f"Dataset records : {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    print(f"\nAccuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=[
            "Likely to Stay",
            "Likely to Leave"
        ],
        zero_division=0
    ))

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }

    # Reference values for individual explanations
    reference_values = {}

    for feature in FEATURES:

        if feature in numerical_features:
            reference_values[feature] = float(
                X_train[feature].median()
            )
        else:
            modes = X_train[feature].mode(dropna=True)

            reference_values[feature] = (
                modes.iloc[0] if not modes.empty else None
            )

    # Retrain using the complete dataset
    model.fit(X, y)

    # Save complete model package
    joblib.dump({
        "model": model,
        "accuracy": metrics["accuracy"],
        "metrics": metrics,
        "features": FEATURES,
        "reference_values": reference_values,
        "version": 2
    }, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")

    return model, metrics["accuracy"]


# =====================================================
# MODEL LOADING
# =====================================================

def _load_artifact():

    if not os.path.exists(MODEL_PATH):
        train_model()

    saved_data = joblib.load(MODEL_PATH)

    # Support older ATTRISENSE model files
    if not isinstance(saved_data, dict) or "model" not in saved_data:
        train_model()
        saved_data = joblib.load(MODEL_PATH)

    return saved_data


def load_model():

    saved_data = _load_artifact()

    return (
        saved_data["model"],
        saved_data.get("accuracy", 0)
    )


# =====================================================
# RISK CLASSIFICATION
# =====================================================

def classify_risk(probability):

    if probability < LOW_RISK_LIMIT:
        return "Low"

    elif probability < MEDIUM_RISK_LIMIT:
        return "Medium"

    return "High"


# =====================================================
# PROBABILITY CALCULATION
# =====================================================

def _get_leave_probability(model, employee_df):

    probabilities = model.predict_proba(employee_df)[0]

    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    leave_index = classes.index(1)

    return float(probabilities[leave_index]) * 100


# =====================================================
# EXPLAINABLE PREDICTION
# =====================================================

def explain_prediction(
    model,
    employee_df,
    reference_values
):

    original_probability = _get_leave_probability(
        model,
        employee_df
    )

    explanations = []

    # Change one feature at a time to its training reference.
    # This measures model sensitivity, not causation.

    for feature in FEATURES:

        reference_value = reference_values.get(feature)

        if reference_value is None:
            continue

        modified_df = employee_df.copy()

        modified_df.loc[0, feature] = reference_value

        modified_probability = _get_leave_probability(
            model,
            modified_df
        )

        change = (
            original_probability - modified_probability
        )

        explanations.append({
            "feature": feature,
            "current_value": str(employee_df.iloc[0][feature]),
            "reference_value": str(reference_value),
            "risk_change": round(change, 2),
            "effect": (
                "Increases risk"
                if change > 0
                else "Decreases risk"
                if change < 0
                else "Little change"
            )
        })

    # Show the strongest positive risk contributions first
    explanations.sort(
        key=lambda item: item["risk_change"],
        reverse=True
    )

    positive_factors = [
        item for item in explanations
        if item["risk_change"] > 0.5
    ]

    if positive_factors:
        return positive_factors[:5]

    # If no positive contributors exist, show the largest
    # model sensitivities without implying increased risk.
    explanations.sort(
        key=lambda item: abs(item["risk_change"]),
        reverse=True
    )

    return explanations[:3]


# =====================================================
# EMPLOYEE RETENTION RECOMMENDATIONS
# =====================================================

def generate_recommendations(employee_data):

    recommendations = []

    overtime = str(
        employee_data.get("OverTime", "")
    ).strip().lower()

    if overtime == "yes":
        recommendations.append(
            "Review overtime hours and consider balancing the employee's workload."
        )

    try:
        if float(employee_data.get("JobSatisfaction", 5)) <= 2:
            recommendations.append(
                "Arrange a discussion to understand job satisfaction concerns."
            )
    except (ValueError, TypeError):
        pass

    try:
        if float(employee_data.get("WorkLifeBalance", 5)) <= 2:
            recommendations.append(
                "Consider flexible working arrangements and better work-life balance."
            )
    except (ValueError, TypeError):
        pass

    try:
        if float(employee_data.get("EnvironmentSatisfaction", 5)) <= 2:
            recommendations.append(
                "Review the workplace environment and address employee concerns."
            )
    except (ValueError, TypeError):
        pass

    try:
        if float(employee_data.get("YearsAtCompany", 0)) <= 1:
            recommendations.append(
                "Provide onboarding support, mentoring and regular feedback."
            )
    except (ValueError, TypeError):
        pass

    try:
        if float(employee_data.get("DistanceFromHome", 0)) >= 20:
            recommendations.append(
                "Explore flexible schedules or commuting support."
            )
    except (ValueError, TypeError):
        pass

    if not recommendations:
        recommendations.append(
            "Continue regular employee engagement, feedback and career development discussions."
        )

    return recommendations


# =====================================================
# MAIN EMPLOYEE PREDICTION FUNCTION
# =====================================================

def predict_employee(employee_data):

    saved_data = _load_artifact()

    model = saved_data["model"]
    accuracy = saved_data.get("accuracy", 0)

    # Validate required input fields
    missing_fields = [
        feature for feature in FEATURES
        if feature not in employee_data
        or employee_data[feature] is None
        or str(employee_data[feature]).strip() == ""
    ]

    if missing_fields:
        raise ValueError(
            f"Missing employee fields: {', '.join(missing_fields)}"
        )

    employee_df = pd.DataFrame(
        [{feature: employee_data[feature] for feature in FEATURES}],
        columns=FEATURES
    )

    prediction = int(model.predict(employee_df)[0])

    risk_probability = _get_leave_probability(
        model,
        employee_df
    )

    risk_level = classify_risk(risk_probability)

    result = (
        "Likely to Leave"
        if prediction == 1
        else "Likely to Stay"
    )

    reference_values = saved_data.get("reference_values", {})

    # Backward compatibility for the previous model artifact
    if not reference_values:
        download_dataset()
        training_data = pd.read_csv(DATA_PATH)

        reference_values = {}

        for feature in FEATURES:
            if feature in training_data.select_dtypes(
                include=["object"]
            ).columns:
                modes = training_data[feature].mode(dropna=True)
                reference_values[feature] = (
                    modes.iloc[0] if not modes.empty else None
                )
            else:
                reference_values[feature] = float(
                    training_data[feature].median()
                )

    risk_factors = explain_prediction(
        model,
        employee_df,
        reference_values
    )

    recommendations = generate_recommendations(
        employee_data
    )

    metrics = saved_data.get("metrics", {
        "accuracy": round(float(accuracy), 2)
    })

    return {
        # Original fields - preserved for existing Flask code
        "prediction": result,
        "attrition_probability": round(risk_probability, 2),
        "model_accuracy": round(float(accuracy), 2),

        # New ATTRISENSE features
        "risk_percentage": round(risk_probability, 2),
        "risk_level": risk_level,
        "risk_category": risk_level,
        "risk_factors": risk_factors,
        "retention_recommendations": recommendations,
        "model_metrics": metrics
    }


# =====================================================
# RUN MODEL TRAINING
# =====================================================

if __name__ == "__main__":
    train_model()
