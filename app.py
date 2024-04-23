from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('next_day_price_model.pkl')

# Load the fitted preprocessor
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    market_name = request.form['market_name']
    grain = request.form['grain']
    variety = request.form['variety']
    min_price = float(request.form['min_price'])
    max_price = float(request.form['max_price'])

    # Preprocess input data
    input_data = pd.DataFrame({
        'Market Name': [market_name],
        'Grain': [grain],
        'Variety': [variety],
        'Min Price (Rs./Quintal)': [min_price],
        'Max Price (Rs./Quintal)': [max_price]
    })
    input_data_encoded = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_encoded)

    # Return prediction as JSON response
    return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)

