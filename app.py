from flask import Flask, jsonify, request, render_template
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the loan eligibility data from the request
    data = request.get_json()

    # Perform prediction using the loaded model
    prediction = model.predict(data)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

