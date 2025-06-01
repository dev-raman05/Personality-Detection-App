from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import traceback
import joblib
from ohe_data import final_data_prep

app = Flask(__name__)

final_data = joblib.load("data_preparation_function.pkl")
model = joblib.load("personality_prediction_model.pkl")
columns_data = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']
train_cols = joblib.load("train_columns.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("homepage.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = []
        for i in range(0, len(columns_data)):
            val = request.form.get(columns_data[i])
            if val is None or val.strip() == "":
                print("Entered value is None or empty for a feature.")
                return render_template(
                    "homepage.html",
                    error=f"Please enter a value for {columns_data[i]}.",
                )
            data.append(val)

        features = pd.DataFrame([data], columns=columns_data)

        numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency']
        
        for nc in numeric_cols:
            features[nc] = features[nc].astype(int)

        features_ohe = final_data(features)
        features_aligned = features_ohe.reindex(columns=train_cols, fill_value=0)

        prediction = model.predict(features_aligned)[0]

        return render_template("homepage.html", prediction=prediction)

    except Exception as e:
        app.logger.error(f"Error in /predict: {e}")
        
        app.logger.error(traceback.format_exc())

        return render_template("homepage.html", error="An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
