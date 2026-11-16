
import os
from io import BytesIO
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify,
    abort, redirect, url_for, send_file
)

from flask_login import login_required, current_user
from sqlalchemy import func
from werkzeug.exceptions import HTTPException

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from ml_model import predict_employee
from extensions import db, login_manager
from database.models import User, Prediction
from auth.routes import auth


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

os.makedirs(app.instance_path, exist_ok=True)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "development-key-change-before-deployment"
)

database_path = os.path.join(
    app.instance_path,
    "attrisense.db"
).replace("\\", "/")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + database_path
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# ==========================================
# INITIALIZE EXTENSIONS
# ==========================================

db.init_app(app)
login_manager.init_app(app)

app.register_blueprint(auth)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _is_attrition_prediction(record):

    label = str(record.prediction or "").strip().lower()

    if label in {
        "yes", "1", "true", "left", "attrition",
        "high risk", "high-risk", "likely to leave"
    }:
        return True

    if label in {
        "no", "0", "false", "stayed", "stay",
        "low risk", "low-risk", "likely to stay"
    }:
        return False

    try:
        probability = float(
            record.attrition_probability or 0
        )

        if 0 <= probability <= 1:
            probability *= 100

        return probability >= 50

    except (TypeError, ValueError):
        return False


def _average_income(records):

    values = []

    for record in records:

        try:
            value = float(
                (record.employee_data or {}).get(
                    "MonthlyIncome", 0
                )
            )

            if value > 0:
                values.append(value)

        except (TypeError, ValueError):
            continue

    return round(
        sum(values) / len(values), 2
    ) if values else 0


def _get_dashboard_data(records):

    left_records = [
        record for record in records
        if _is_attrition_prediction(record)
    ]

    stayed_records = [
        record for record in records
        if not _is_attrition_prediction(record)
    ]

    total = len(records)
    left = len(left_records)
    stayed = len(stayed_records)

    attrition_rate = (
        round((left / total) * 100, 2)
        if total else 0
    )

    departments = {
    "Sales": 0,
    "Research & Development": 0,
    "Human Resources": 0
}

    age_groups = {
        "18-25": 0,
        "26-35": 0,
        "36-45": 0,
        "46-55": 0,
        "56+": 0
    }

    overtime = {
        "Yes": 0,
        "No": 0
    }

    satisfaction = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
        "Very High": 0
    }

    satisfaction_map = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }

    for record in left_records:

        data = record.employee_data or {}

        department = str(
            data.get("Department") or "Unknown"
        )

        departments[department] = (
            departments.get(department, 0) + 1
        )

        try:
            age = int(data.get("Age", 0))

            if 18 <= age <= 25:
                age_groups["18-25"] += 1
            elif 26 <= age <= 35:
                age_groups["26-35"] += 1
            elif 36 <= age <= 45:
                age_groups["36-45"] += 1
            elif 46 <= age <= 55:
                age_groups["46-55"] += 1
            elif age >= 56:
                age_groups["56+"] += 1

        except (TypeError, ValueError):
            pass

        overtime_value = str(
            data.get("OverTime") or ""
        )

        if overtime_value in overtime:
            overtime[overtime_value] += 1

        try:
            score = int(
                data.get("JobSatisfaction", 0)
            )

            satisfaction_label = satisfaction_map.get(
                score
            )

            if satisfaction_label:
                satisfaction[satisfaction_label] += 1

        except (TypeError, ValueError):
            pass

    stayed_income = _average_income(stayed_records)
    left_income = _average_income(left_records)

    outcome_counts = {}

    for record in records:

        label = str(
            record.prediction or "Unknown"
        ).strip() or "Unknown"

        outcome_counts[label] = (
            outcome_counts.get(label, 0) + 1
        )

    return {
        "total": total,
        "left": left,
        "stayed": stayed,
        "attrition_rate": attrition_rate,

        "departments": {
            "labels": list(departments.keys()),
            "values": list(departments.values())
        },

        "age": {
            "labels": list(age_groups.keys()),
            "values": list(age_groups.values())
        },

        "age_groups": {
            "labels": list(age_groups.keys()),
            "values": list(age_groups.values())
        },

        "overtime": {
            "labels": list(overtime.keys()),
            "values": list(overtime.values())
        },

        "satisfaction": {
            "labels": list(satisfaction.keys()),
            "values": list(satisfaction.values())
        },

        "income": {
            "stayed": stayed_income,
            "left": left_income
        },

        "average_income": {
            "labels": ["Stayed", "Left"],
            "values": [stayed_income, left_income]
        },

        "outcomes": {
            "labels": list(outcome_counts.keys()),
            "values": list(outcome_counts.values())
        }
    }


# ==========================================
# OLD ADMIN LOGIN REDIRECT
# ==========================================

@app.route("/admin/login")
def old_admin_login():

    return redirect(
        url_for("auth.login", role="admin")
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
@login_required
def home():

    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))

    if current_user.role != "user":
        abort(403)

    return render_template("index.html")


# ==========================================
# EMPLOYEE ATTRITION PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@login_required
def predict_employee_route():

    if current_user.role != "user":

        return jsonify({
            "success": False,
            "error": "Only normal users can make predictions."
        }), 403

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({
                "success": False,
                "error": "Please submit valid employee information."
            }), 400

        employee_name = str(
            data.get("EmployeeName", "")
        ).strip()

        employee_id = str(
            data.get("EmployeeID", "")
        ).strip()

        if not employee_name or not employee_id:

            return jsonify({
                "success": False,
                "error": "Employee name and employee ID are required."
            }), 400

        if len(employee_name) > 100 or len(employee_id) > 50:

            return jsonify({
                "success": False,
                "error": "Employee name or ID exceeds the allowed length."
            }), 400

        existing_employee = Prediction.query.filter(
            Prediction.user_id == current_user.id,
            func.lower(Prediction.employee_id) == employee_id.lower()
        ).first()

        if existing_employee:

            return jsonify({
                "success": False,
                "error": "This Employee ID already exists in your account."
            }), 409

        employee_data = {
            "Age": int(data["Age"]),
            "BusinessTravel": data["BusinessTravel"],
            "Department": data["Department"],
            "DistanceFromHome": int(data["DistanceFromHome"]),
            "JobRole": data["JobRole"],
            "JobSatisfaction": int(data["JobSatisfaction"]),
            "MaritalStatus": data["MaritalStatus"],
            "MonthlyIncome": int(data["MonthlyIncome"]),
            "OverTime": data["OverTime"],
            "TotalWorkingYears": int(data["TotalWorkingYears"]),
            "YearsAtCompany": int(data["YearsAtCompany"]),
            "WorkLifeBalance": int(data["WorkLifeBalance"]),
            "EnvironmentSatisfaction": int(
                data["EnvironmentSatisfaction"]
            )
        }

        # Basic numeric validation
        if not 18 <= employee_data["Age"] <= 70:
            raise ValueError("Age must be between 18 and 70.")

        if employee_data["DistanceFromHome"] < 0:
            raise ValueError("Distance from home cannot be negative.")

        if employee_data["MonthlyIncome"] <= 0:
            raise ValueError("Monthly income must be greater than zero.")

        if employee_data["TotalWorkingYears"] < 0:
            raise ValueError("Total working years cannot be negative.")

        if employee_data["YearsAtCompany"] < 0:
            raise ValueError("Years at company cannot be negative.")

        if employee_data["JobSatisfaction"] not in [1, 2, 3, 4]:
            raise ValueError("Invalid job satisfaction value.")

        if employee_data["WorkLifeBalance"] not in [1, 2, 3, 4]:
            raise ValueError("Invalid work-life balance value.")

        if employee_data["EnvironmentSatisfaction"] not in [1, 2, 3, 4]:
            raise ValueError("Invalid environment satisfaction value.")

        if employee_data["OverTime"] not in ["Yes", "No"]:
            raise ValueError("Invalid overtime value.")

        # Generate prediction
        result = predict_employee(employee_data)

        # Save prediction with the logged-in user's ID
        prediction_record = Prediction(
            user_id=current_user.id,
            employee_name=employee_name,
            employee_id=employee_id,
            employee_data=employee_data,
            prediction=result["prediction"],
            attrition_probability=result["attrition_probability"],
            model_accuracy=result["model_accuracy"]
        )

        db.session.add(prediction_record)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Your data has been added successfully!",
            "result": result,
            "employee": {
                "name": employee_name,
                "id": employee_id
            }
        })

    except (KeyError, TypeError, ValueError) as error:

        db.session.rollback()

        return jsonify({
            "success": False,
            "error": f"Invalid employee information: {error}"
        }), 400

    except Exception:

        db.session.rollback()
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Prediction could not be completed."
        }), 500


# ==========================================
# DASHBOARD ANALYTICS API
# ==========================================

@app.route("/api/dashboard")
@login_required
def dashboard_analytics():

    try:

        if current_user.role == "admin":

            records = Prediction.query.order_by(
                Prediction.created_at.asc()
            ).all()

        elif current_user.role == "user":

            records = Prediction.query.filter_by(
                user_id=current_user.id
            ).order_by(
                Prediction.created_at.asc()
            ).all()

        else:
            abort(403)

        return jsonify(
            _get_dashboard_data(records)
        )

    except HTTPException:
        raise

    except Exception:

        app.logger.exception(
            "Dashboard analytics failed"
        )

        return jsonify({
            "error": "Unable to load dashboard analytics."
        }), 500


# ==========================================
# PERSONAL PREDICTION HISTORY
# ==========================================

@app.route("/history")
@login_required
def history():

    # Admins use the admin dashboard
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))

    if current_user.role != "user":
        abort(403)

    # IMPORTANT:
    # Only retrieve records owned by the logged-in user.
    predictions = (
        Prediction.query
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return render_template(
        "history.html",
        predictions=predictions
    )


# ==========================================
# ADMIN DASHBOARD
# ==========================================

@app.route("/admin")
@login_required
def admin_dashboard():

    if current_user.role != "admin":
        abort(403)

    total_users = User.query.filter_by(
        role="user"
    ).count()

    total_predictions = Prediction.query.count()

    users = User.query.order_by(
        User.created_at.desc()
    ).all()

    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        users=users,
        predictions=predictions
    )


# ==========================================
# ADMIN EXCEL REPORT EXPORT
# ==========================================

@app.route("/admin/export-excel")
@login_required
def export_excel():

    if current_user.role != "admin":
        abort(403)

    try:

        workbook = Workbook()

        # EMPLOYEE PREDICTIONS SHEET
        prediction_sheet = workbook.active
        prediction_sheet.title = "Employee Predictions"

        prediction_sheet.append([
            "Prediction ID",
            "Employee Name",
            "Employee ID",
            "Submitted By",
            "Email",
            "Age",
            "Department",
            "Job Role",
            "Monthly Income",
            "Business Travel",
            "Distance From Home",
            "Job Satisfaction",
            "Marital Status",
            "Overtime",
            "Total Working Years",
            "Years At Company",
            "Work Life Balance",
            "Environment Satisfaction",
            "Attrition Prediction",
            "Attrition Probability (%)",
            "Model Accuracy (%)",
            "Date"
        ])

        predictions = Prediction.query.order_by(
            Prediction.created_at.desc()
        ).all()

        for record in predictions:

            employee_data = record.employee_data or {}

            probability = float(
                record.attrition_probability or 0
            )

            accuracy = float(
                record.model_accuracy or 0
            )

            if 0 <= probability <= 1:
                probability *= 100

            if 0 <= accuracy <= 1:
                accuracy *= 100

            prediction_sheet.append([
                record.id,
                record.employee_name,
                record.employee_id,
                record.user.name if record.user else "",
                record.user.email if record.user else "",
                employee_data.get("Age", ""),
                employee_data.get("Department", ""),
                employee_data.get("JobRole", ""),
                employee_data.get("MonthlyIncome", ""),
                employee_data.get("BusinessTravel", ""),
                employee_data.get("DistanceFromHome", ""),
                employee_data.get("JobSatisfaction", ""),
                employee_data.get("MaritalStatus", ""),
                employee_data.get("OverTime", ""),
                employee_data.get("TotalWorkingYears", ""),
                employee_data.get("YearsAtCompany", ""),
                employee_data.get("WorkLifeBalance", ""),
                employee_data.get("EnvironmentSatisfaction", ""),
                record.prediction,
                round(probability, 2),
                round(accuracy, 2),
                record.created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ) if record.created_at else ""
            ])

        # REGISTERED USERS SHEET
        user_sheet = workbook.create_sheet(
            "Registered Users"
        )

        user_sheet.append([
            "User ID",
            "Name",
            "Email",
            "Role",
            "Registration Date"
        ])

        users = User.query.order_by(
            User.created_at.desc()
        ).all()

        for user in users:

            user_sheet.append([
                user.id,
                user.name,
                user.email,
                user.role,
                user.created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ) if user.created_at else ""
            ])

        # SUMMARY SHEET
        summary_sheet = workbook.create_sheet("Summary")

        total_users = User.query.filter_by(
            role="user"
        ).count()

        summary_sheet.append(["ATTRISENSE REPORT"])

        summary_sheet.append([
            "Generated On",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

        summary_sheet.append([])
        summary_sheet.append(["Metric", "Value"])

        summary_sheet.append([
            "Registered Users",
            total_users
        ])

        summary_sheet.append([
            "Total Predictions",
            len(predictions)
        ])

        prediction_counts = {}

        for record in predictions:

            label = str(
                record.prediction or "Unknown"
            ).strip()

            prediction_counts[label] = (
                prediction_counts.get(label, 0) + 1
            )

        summary_sheet.append([])
        summary_sheet.append([
            "Prediction Result",
            "Count"
        ])

        for label, count in sorted(
            prediction_counts.items()
        ):
            summary_sheet.append([
                label,
                count
            ])

        # FORMAT WORKSHEETS
        header_fill = PatternFill(
            fill_type="solid",
            fgColor="4338CA"
        )

        header_font = Font(
            bold=True,
            color="FFFFFF"
        )

        for sheet in workbook.worksheets:

            sheet.freeze_panes = "A2"

            header_rows = (
                [1, 4, 9]
                if sheet.title == "Summary"
                else [1]
            )

            for row_number in header_rows:

                for cell in sheet[row_number]:

                    cell.font = header_font
                    cell.fill = header_fill

                    cell.alignment = Alignment(
                        horizontal="center",
                        vertical="center"
                    )

            for column_number in range(
                1, sheet.max_column + 1
            ):

                max_length = 0

                for row_number in range(
                    1, sheet.max_row + 1
                ):

                    cell_value = sheet.cell(
                        row=row_number,
                        column=column_number
                    ).value

                    if cell_value is not None:

                        max_length = max(
                            max_length,
                            len(str(cell_value))
                        )

                column_letter = get_column_letter(
                    column_number
                )

                sheet.column_dimensions[
                    column_letter
                ].width = min(
                    max_length + 4, 40
                )

        # CREATE EXCEL DOWNLOAD
        output = BytesIO()

        workbook.save(output)
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="ATTRISENSE_Report.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    except Exception:

        app.logger.exception(
            "Excel export failed"
        )

        return jsonify({
            "success": False,
            "error": "Unable to generate Excel report."
        }), 500


# ==========================================
# CREATE DATABASE TABLES
# ==========================================

with app.app_context():
    db.create_all()


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
