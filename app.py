from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model using pickle
with open("calorie_burn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form in index.html
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        temp = float(request.form['temp'])
        
        # Organize input data as per the model's expected format
        input_data = np.array([[gender, age, height, weight, duration, heart_rate, temp]])

        # Predict the calorie burn
        prediction = model.predict(input_data)

        # Render the result back to index.html
        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        # If there's an error, display it on the page
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
