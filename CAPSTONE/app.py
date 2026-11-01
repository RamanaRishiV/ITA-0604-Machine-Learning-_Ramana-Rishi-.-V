
from flask import Flask, render_template, request, jsonify
from ml_model import predict_employee

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

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

        result = predict_employee(employee_data)

        return jsonify({
            "success": True,
            "result": result
        })

    except (KeyError, TypeError, ValueError) as error:
        return jsonify({
            "success": False,
            "error": f"Invalid employee information: {error}"
        }), 400

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "success": False,
            "error": "Prediction could not be completed."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
