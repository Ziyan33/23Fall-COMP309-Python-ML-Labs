# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the models
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)

# Loading the label encoders
with open('bike_make_encoder.pkl', 'rb') as file:
    bike_make_encoder = pickle.load(file)

with open('bike_model_encoder.pkl', 'rb') as file:
    bike_model_encoder = pickle.load(file)

@app.route('/')
def index():
    return render_template('predict_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and process form data
        primary_offence = int(request.form['primary_offence'])
        

        # Retrieve and transform 'BIKE_MAKE' and 'BIKE_MODEL'
        bike_make = request.form['BIKE_MAKE']
        bike_model = request.form['BIKE_MODEL']        
        bike_make_encoded = bike_make_encoder.transform([bike_make])[0]
        bike_model_encoded = bike_model_encoder.transform([bike_model])[0]      
        
        bike_cost_str = request.form['bike_cost'].replace('$', '').replace(',', '')
        bike_cost = float(bike_cost_str) if bike_cost_str else 0
        report_doy = int(request.form['report_doy'])

        # Make predictions using both models
        prediction_decision_tree = decision_tree_model.predict([[primary_offence, bike_cost, report_doy, bike_make_encoded, bike_model_encoded]])[0]
        prediction_logistic_regression = logistic_regression_model.predict([[primary_offence, bike_cost, report_doy, bike_make_encoded, bike_model_encoded]])[0]

        # Render result template with the predictions
        return render_template('result.html', prediction_dt=prediction_decision_tree, prediction_lr=prediction_logistic_regression)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
