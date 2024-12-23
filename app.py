from flask import Flask, request, render_template # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

# Load the saved model
model = joblib.load('mental_health_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Route for home page (Form for input)
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    gender = 1 if request.form['gender'] == 'Female' else 0  # Encoding gender
    family_history = 1 if request.form['family_history'] == 'Yes' else 0  # Encoding family history
    anxiety=1 if request.form['anxiety']=='Yes' else 0
    sleep=1 if request.form['sleep']=='Regular' else 0
    
    # Prepare data for prediction
    input_data = np.array([[age, gender, family_history,anxiety,sleep]])
    
    # Use the model to predict
    prediction = model.predict(input_data)[0]  # Get the predicted class (0 or 1)
    
    # Translate prediction into readable output
    if prediction == 1:
        result = "Likely to have mental health issues."
    else:
        result = "Unlikely to have mental health issues."
    
    # Render the result page
    return render_template('result.html', prediction_text=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
