import pickle
from flask import Flask, request, jsonify

model_file = 'model.bin'

with open(model_file, 'rb')  as f_in:
    dv, model = pickle.load(f_in)

app = Flask('placement')

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()

    X = dv.transform([student])
    y_pred = model.predict_proba(X)[0,1]
    placement = y_pred > 0.5

    result = {
        "placement_probability": float(y_pred),
        "placement": bool(placement)
    }

    return jsonify(result)

if __name__ == "__main__":
    # app.run(debug=True, host='122.173.228.5', port=9696)
    app.run(debug=True, host='0.0.0.0', port=9696)