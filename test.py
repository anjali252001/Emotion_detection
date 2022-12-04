import speech_recognition as sr
import numpy as np
import requests
import json


r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak Anything :")
    audio = r.listen(source)
    try:  
        X_test= r.recognize_google(audio)
        X_test= format(X_test)
        payload = {'data': X_test}
        print(payload)
        y_predict = requests.post('http://127.0.0.1:5000/emotion-classification', json=payload).text
        print(y_predict)
    except:
        print("sorry")
        
# f = open("try.txt", "r")
# x= f.read()
# x= x.split(":",1)
# X_test= x[1]
# payload = {'data': X_test}
# print(payload)
# y_predict = requests.post('http://127.0.0.1:5000/emotion-classification', json=payload).text
# print(y_predict)


