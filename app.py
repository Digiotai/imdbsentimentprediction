#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import argparse
import time
import uuid
import base64
import os
import pandas as pd
import re
import pickle
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.models import Sequential, load_model,model_from_json
from tensorflow.keras import backend as K 


UPLOAD_FOLDER = 'uploads'
model_path = 'IMDB.h5'


model_weights_path = 'IMDBW.h5'

one  = pickle.load(open('Tokenizer', 'rb'))

Tag_re = re.compile(r'<[^>]+>')

def preprocess_text(text):
    sen = remove_tags(text)
    sen = re.sub('[^a-zA-Z]', ' ', sen)
    sen = re.sub(r"\s+[a-zA-Z]\s+", ' ', sen)
    sen = re.sub(r'\s+', ' ', sen)
    return sen
def remove_tags(text):
    return Tag_re.sub('',text)
def predict1(text):
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    model.load_weights(model_weights_path)

    user=preprocess_text(text)
    user = one.texts_to_sequences(user)
    z= []
    for i in user:
      for y in i:    
        z.append(y)
    z = [z]
    user = pad_sequences(z,padding='post',maxlen=100)
    res=int(model.predict(user))
    keras.backend.clear_session()
    print(res)
    return res
		



def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Upload")
        mytext = request.form['text']
        
        res=predict1(mytext)
        if res==1:
           msg='positive'
        else:
           msg='negative'

        return render_template('template.html',label=msg )
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)