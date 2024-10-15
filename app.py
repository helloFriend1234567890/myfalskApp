# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# import os  # Import os to access environment variables

# app = Flask(__name__)

# # Load your models
# with open('models.pkl', 'rb') as f:
#     models = pickle.load(f)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # Get JSON data from request
#     input_data = np.array([data['sex'], data['age'], data['height'], data['weight'],
#                            data['hypertension'], data['diabetes'], data['bmi'],
#                            data['level'], data['fitness_goal'], data['fitness_type']]).reshape(1, -1)

#     predictions = {}
#     for key, model in models.items():
#         predictions[key] = model.predict(input_data)[0]

#     return jsonify(predictions)

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))  # Get the port from environment variable or use 5000 as default
#     app.run(host='0.0.0.0', port=port, debug=True)  # Bind to 0.0.0.0 and use the port


from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

# Load the trained model
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Initialize label encoders for target columns
target_columns = ['Workout Plan', 'Equipment', 'Breakfast', 'Lunch', 'Dinner', 'Snacks']
label_encoders = {}

# Load or create label encoders
for col in target_columns:
    # You can replace this with your own method of creating encoders, if you need to
    label_encoders[col] = LabelEncoder()
    # Add a sample fitting based on the unique values in your dataset
    # (Replace with actual unique values from your training data if necessary)
    unique_values = ["example_value1", "example_value2"]  # Replace with actual values
    label_encoders[col].fit(unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions based on user input."""
    data = request.json

    try:
        # Preprocess input data
        gender = 1 if data['gender'].lower() == 'male' else 0
        fitness_type = data['fitness_type'].lower()
        age = data['age']
        weight = data['weight']
        height = data['height']
        hypertension = 1 if data['hypertension'].lower() == 'yes' else 0
        diabetes = 1 if data['diabetes'].lower() == 'yes' else 0
        bmi = weight / ((height / 100) ** 2)

        user_input = pd.DataFrame([[gender, fitness_type, age, weight, height, hypertension, diabetes, bmi]],
                                   columns=['Gender_Male', 'Fitness Type', 'Age', 'Weight', 'Height', 'Hypertension', 'Diabetes', 'BMI'])

        predicted_recommendations = model.predict(user_input)

        decoded_recommendations = {}
        for idx, col in enumerate(target_columns):
            decoded_recommendations[col] = label_encoders[col].inverse_transform([int(predicted_recommendations[0][idx])])[0]

        return jsonify(decoded_recommendations), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment variable
    app.run(host='0.0.0.0', port=port)  # Bind to 0.0.0.0 and use the environment port
