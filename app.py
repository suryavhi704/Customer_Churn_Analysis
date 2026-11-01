
from flask import Flask, render_template, request
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# Create Flask application
app = Flask(__name__)

# Load the trained model and preprocessing pipeline
model = joblib.load("churn_prediction_model.pkl")


# Home page and prediction route
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    # Run when the user submits the form
    if request.method == "POST":
        try:
            # Read customer information from the form
            customer = {
                "CreditScore": int(request.form["CreditScore"]),
                "Geography": request.form["Geography"],
                "Gender": request.form["Gender"],
                "Age": int(request.form["Age"]),
                "Tenure": int(request.form["Tenure"]),
                "Balance": float(request.form["Balance"]),
                "NumOfProducts": int(request.form["NumOfProducts"]),
                "HasCrCard": int(request.form["HasCrCard"]),
                "IsActiveMember": int(request.form["IsActiveMember"]),
                "EstimatedSalary": float(
                    request.form["EstimatedSalary"]
                )
            }

            # Match the feature names used during training
            customer_data = pd.DataFrame([customer])

            # Validate the age range used in your notebook
            if not 18 <= customer["Age"] <= 65:
                raise ValueError("Age must be between 18 and 65.")

            # Predict: 1 = Exit, 0 = Stay
            prediction = model.predict(customer_data)[0]

            if prediction == 1:
                result = "Customer is likely to EXIT."
            else:
                result = "Customer is likely to STAY."

        except (ValueError, KeyError):
            error = (
                "Please enter valid customer details. "
                "Age must be between 18 and 65."
            )

    # Display the page with the result
    return render_template(
        "index.html",
        result=result,
        error=error
    )


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
