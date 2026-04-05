from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_folder='statics')

model = tf.keras.models.load_model('my_model.keras')
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
class_labels = ['Setosa', 'Versicolor', 'Virginica']

_KEYS = ('sepal_length', 'sepal_width', 'petal_length', 'petal_width')


def _parse_features():
    """Return (sl, sw, pl, pw) from JSON, form, or query args; None if incomplete."""
    if request.method == 'GET':
        if not all(k in request.args for k in _KEYS):
            return None
        src = request.args
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        if not all(k in body for k in _KEYS):
            return None
        src = body
    else:
        if not request.form or not all(k in request.form for k in _KEYS):
            return None
        src = request.form
    return tuple(float(src[k]) for k in _KEYS)


def _features_array(features):
    return np.array(features, dtype=np.float64).reshape(1, -1)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/')
def home():
    return 'Welcome to the Iris Classifier API!'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    features = _parse_features()
    if features is None:
        if request.method == 'GET':
            return render_template('predict.html')
        return jsonify({'error': 'Missing one or more of: ' + ', '.join(_KEYS)}), 400

    x_scaled = scaler.transform(_features_array(features))
    pred = model.predict(x_scaled, verbose=0)
    predicted_class = class_labels[int(np.argmax(pred))]
    return jsonify({'predicted_class': predicted_class, 'model': 'keras'})


@app.route('/predict_knn', methods=['GET', 'POST'])
def predict_knn():
    features = _parse_features()
    if features is None:
        if request.method == 'GET':
            return render_template('predict_knn.html')
        return jsonify({'error': 'Missing one or more of: ' + ', '.join(_KEYS)}), 400

    x_scaled = scaler.transform(_features_array(features))
    label_idx = int(knn_model.predict(x_scaled)[0])
    predicted_class = class_labels[label_idx]
    return jsonify({'predicted_class': predicted_class, 'model': 'knn'})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
