
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('home.html')  # This renders the homepage with information

## Route for prediction form and data submission
@app.route('/predict', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('index.html')  # Renders the input form page
    else:
        # Collect the molecular descriptor inputs from the form
        data = CustomData(
            CIC0=float(request.form.get('CIC0')),
            SM1_DzZ=float(request.form.get('SM1_DzZ')),
            GATS1i=float(request.form.get('GATS1i')),
            NdsCH=int(request.form.get('NdsCH')),
            NdssC=int(request.form.get('NdssC')),
            MLOGP=float(request.form.get('MLOGP'))
        )


        # Convert input data into DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        # Use the prediction pipeline to generate predictions
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction Result:", results)

        # Render the result back to the form page with the results
        return render_template('index.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
