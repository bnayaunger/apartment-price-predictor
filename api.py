

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import joblib
from assets_data_prep import prepare_data


app = Flask(__name__)
data = joblib.load("ElasticNet_model_and_columns.pkl")
model = data['model']
expected_columns = data['columns']
features = joblib.load("all_params.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'property_type' : request.form['property_type'],
        'room_num': float(request.form['room_num']),
        'area': float(request.form['area']),
        'floor': float(request.form['floor']),
        'neighborhood': request.form['neighborhood']
        }
    total_floors_str = request.form['total_floors']
    data['total_floors'] = float(total_floors_str) if total_floors_str.strip() else np.nan
    checkbox_features = [
        'ac', 'handicap', 'is_renovated', 'has_balcony', 'has_bars', 'elevator',
        'has_safe_room', 'is_furnished', 'has_storage', 'has_parking'
    ]

    for feature in checkbox_features:
        data[feature] = feature in request.form
   
    # בדיקה: אם סוג הנכס הוא פנטהאוז, הקומה חייבת להיות שווה לסך הקומות
    if data['property_type'] == 'פנטהאוז':
        if not pd.isna(data['total_floors']) and data['floor'] != data['total_floors']:
            return render_template('index.html', prediction=None, error="בפנטהאוז הקומה חייבת להיות הקומה האחרונה.", form_data=request.form)
    if not pd.isna(data['total_floors']) and data['floor'] > data['total_floors']:
        return render_template('index.html', prediction=None, error="קומה לא יכולה להיות גבוהה ממספר הקומות הכולל.", form_data=request.form)

    df = pd.DataFrame([data])
    df_prepared = prepare_data(df) 
    df_prepared = df_prepared.reindex(columns=expected_columns, fill_value=0)
    prediction = round(model.predict(df_prepared)[0], 2)

    return render_template('index.html', prediction=prediction, form_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
