import os
import sys
import shutil
import time

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import urllib.request
import json
from geopy.geocoders import Nominatim

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/images/<Start>')
def download_file(Start):
    return send_from_directory(app.config['images'], Start)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/result', methods=['POST'])
def predict():
    rfc = joblib.load('model/rf_model.joblib')
    print('model loaded')

    if request.method == 'POST':
        address = request.form['Location']
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(address, timeout=None)
        print(location.address)
        lat = [location.latitude]
        log = [location.longitude]
        latlong = pd.DataFrame({'latitude': lat, 'longitude': log})
        print(latlong)

        DT = request.form['timestamp']
        latlong['timestamp'] = DT
        data = latlong
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')
        column_1 = data.iloc[:, 0]
        DT = pd.DataFrame(
            {
                "year": column_1.dt.year,
                "month": column_1.dt.month,
                "day": column_1.dt.day,
                "hour": column_1.dt.hour,
                "dayofyear": column_1.dt.dayofyear,
                "week": column_1.dt.isocalendar().week,
                "dayofweek": column_1.dt.dayofweek,
                "weekday": column_1.dt.weekday,
                "quarter": column_1.dt.quarter,
            }
        )
        data = data.drop('timestamp', axis=1)
        final = pd.concat([DT, data], axis=1)

        if final.shape[1] < 9:
            return "Error: DataFrame doesn't have enough columns."

        X = final.iloc[:, [1, 2, 3, 4, 5, 7, 8]].values
        my_prediction = rfc.predict(X)

        crimes = {
            0: 'Act 379-Robbery',
            1: 'Act 13-Gambling',
            2: 'Act 279-Accident',
            3: 'Act 323-Violence',
            4: 'Act 363-kidnapping',
            5: 'Act 302-Murder'
        }

        predicted_crime_indices = np.where(my_prediction == 1)[0]
        predicted_crime = [crimes[i] for i in predicted_crime_indices]

        if predicted_crime_indices.size > 0:
            my_prediction = f'Predicted crime: {", ".join(predicted_crime)}'
        else:
            my_prediction = 'Place is safe, no crime is expected at that timestamp.'

    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
