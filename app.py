# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os, ssl, uuid
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import InputLayer

# Monkey‐patch InputLayer to accept old 'batch_shape' configs
_orig_input_from_config = InputLayer.from_config
@classmethod
def _patched_from_config(cls, config):
    if 'batch_shape' in config and 'batch_input_shape' not in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    return _orig_input_from_config(config)
InputLayer.from_config = _patched_from_config

# SSL fix for some Windows installs
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Load the Keras model
MODEL_PATH = os.path.join('model', 'model.h5')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    raise


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(path):
    img = load_img(path, target_size=(64,64))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error='No file provided'), 400
    f = request.files['file']
    if not f or f.filename == '' or not allowed_file(f.filename):
        return jsonify(error='Invalid file'), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(fpath)

    try:
        img = preprocess_image(fpath)
        preds = model.predict(img)[0]
        idx = int(np.argmax(preds))
        labels = ['A+','A-','B+','B-','AB+','AB-','O+','O-']
        return jsonify(
            predicted_class=idx,
            predicted_label=labels[idx],
            confidence=float(np.max(preds))
        )
    except Exception as ex:
        return jsonify(error=str(ex)), 500
    finally:
        try: os.remove(fpath)
        except: pass


if __name__ == '__main__':
    app.run(debug=True)