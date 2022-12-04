from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import joblib
import re
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')

stopword_all =  list(set(stopwords.words('english'))) 
stopword_all.extend(list(string.punctuation))

def clean_text(text):
    text = text.lower()
    text= re.sub(r"(#[\d\w\.]+)", '', text)
    text = re.sub(r"(@[\d\w\.]+)", '', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stopword_all]
#     print(text)
    text = ' '.join(text)
    return text

class EmotionClassification(Resource):
    def post(self):
        args = parser.parse_args()
        X = args['data']
        print(X)
        X = clean_text(X)
        seq = tokenizer.texts_to_sequences([X])
        X = pad_sequences(seq, maxlen=500)
        val_res = model.predict(X)
        result =le.inverse_transform([np.argmax(val_res)])
        proba =  np.max(val_res)
        X = (f"{result[0]} : {proba}\n\n")
        return jsonify(X)

api.add_resource( EmotionClassification, '/emotion-classification')

if __name__ == '__main__':
    try:
        tokenizer = joblib.load('./tokenizer.pkl')
    except:
        print("cant load")
    try:  
        le= joblib.load('./label_encoder.pkl')
    except:
        print("cant") 
        
    model = tf.keras.models.load_model('./lstm_simple.h5')
    
    
    app.run(debug=True)
        
