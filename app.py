from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model'

# Load your trained model
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def model_predict(img_path, model):



    newtest=cv2.imread(img_path,1)
    newtest=cv2.resize(newtest,(224,224))

    newtest = np.expand_dims(newtest, axis=0)
    PREDICTED_CLASSES = model.predict_classes(newtest, verbose=1)
    preds = PREDICTED_CLASSES
    os.remove(img_path)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        #format=".jpeg"
        #file_path=file_path+format
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        if(preds==0):
            res=str(preds)
            result = str("High Chances of Covid-19")
            result=res+result
        elif(preds==1):
            res=str(preds)
            result = str("Not Covid")
            result=res+result

        return result
    return None


if __name__ == '__main__':
     #app.run(port=5002, debug=True)


    app.run()
