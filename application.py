import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline

# Create a Flask application
application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def index():
    """
    Render the home page.

    Returns:
        flask.render_template: HTML template for the home page.

    """
    return render_template('index.html')

# Route for predicting data points
@app.route('/', methods=['GET','POST'])
def predict_data_point():
    """
    Predict the edibility of a mushroom based on user input.

    Returns:
        flask.render_template: HTML template with the prediction results.

    """
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        data = CustomData(
            bruises=request.form.get("bruises"),
            gill_spacing=request.form.get("gill-spacing"),
            gill_size=request.form.get("gill-size"),
            gill_color=request.form.get("gill-color"),
            stalk_root=request.form.get("stalk-root"),
            ring_type=request.form.get("ring-type"),
            spore_print_color=request.form.get("spore-print-color")
        )
        
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        
        if results == 0.0:
            answer = "edible"
        else:
            answer = "poisonous"
            
        return render_template('index.html', results=answer)
    
if __name__ == "__main__":
    # Run the Flask application in debug mode
    app.run(debug=True)
