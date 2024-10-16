from flask import Flask, request, jsonify
import pickle
import numpy as np
import os  # Import os to access environment variables

app = Flask(__name__)

# Load your models
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from request
    input_data = np.array([data['sex'], data['age'], data['height'], data['weight'],
                           data['hypertension'], data['diabetes'], data['bmi'],
                           data['level'], data['fitness_goal'], data['fitness_type']]).reshape(1, -1)

    predictions = {}
    for key, model in models.items():
        predictions[key] = model.predict(input_data)[0]

    return jsonify(predictions)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get the port from environment variable or use 5000 as default
    app.run(host='0.0.0.0', port=port, debug=True)  # Bind to 0.0.0.0 and use the port
