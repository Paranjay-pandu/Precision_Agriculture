from flask import Flask, request, render_template, jsonify, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    rf = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        air_temp = float(request.form['air_temperature'])
        air_humidity = float(request.form['air_humidity'])
        wind_speed = float(request.form['wind_speed'])
        solar_radiation = float(request.form['solar_radiation'])
        soil_moisture = float(request.form['soil_moisture'])
        soil_humidity = float(request.form['soil_humidity'])
        
        # Create a DataFrame for the input features
        input_data = pd.DataFrame({
            'Air temperature (C)': [air_temp],
            'Air humidity (%)': [air_humidity],
            'Wind speed (Km/h)': [wind_speed],
            'Solar radiation (W/m2)': [solar_radiation],
            'Soil Moisture': [soil_moisture],
            'Soil Humidity': [soil_humidity]
        })
        
        # Ensure the input data has all the features the model expects
        for col in rf.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match the model's expected order
        input_data = input_data.reindex(columns=rf.feature_names_in_, fill_value=0)
        
        # Predict the water requirement
        prediction = rf.predict(input_data)
        return jsonify({'prediction': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
