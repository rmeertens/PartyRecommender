
import os
import pickle

import re
import json
from keras.preprocessing import sequence
from keras.utils import np_utils
from flask import Flask
from flask import request
import numpy as np
import ssl
from firebase import firebase


import model
import partyprogram_loader

# GLOBALS YOU CAN SET AND PLAY WITH
FILENAME_SAVED_MODEL = 'party_model.h5'
FILENAME_SAVED_DATA = 'data.p'
SAVED_TRAINDATA_DIR = 'traindata'
PARTIJPATH = 'partijprogrammas'
FIREBASE_URL = "https://scenic-reason-844.firebaseio.com"

EMBEDDING_VECTOR_LENGTH = 32
MAX_REVIEW_LENGTH = 20

kerasmodel = None
id_of_word_getter = None
party_names = None

app = Flask(__name__)
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
context.load_cert_chain('/etc/mysecrets/cert.pem', keyfile='/etc/mysecrets/privkey.pem')


myfirebase = firebase.FirebaseApplication(FIREBASE_URL, None)


@app.route('/')
def hello_world():
    return 'This is a test. Go to slash predict to see what is up, World!'

@app.route('/predict')
def predict_party():
    sentence = request.headers.get('predicttext')
    print(request.headers)

    sentence = sentence .replace("\n", " ")

    sentence = re.sub(r'\w*\d\w*', '', sentence ).strip()


    # make lower case
    sentence = sentence .lower()
    if (len(sentence ) > 0):
        words = partyprogram_loader.basic_tokenizer(sentence)
    else:
        words = []



    firebasepostresult = myfirebase.post('/sentences', sentence)

    ids = []
    sentence_understood = ""
    for word in words:
        ids.append(id_of_word_getter.get_id_of_word(word))
        sentence_understood += id_of_word_getter.get_word_of_id(id_of_word_getter.get_id_of_word(word)) + " "
    ids = [ids]
    something = sequence.pad_sequences(ids, maxlen=MAX_REVIEW_LENGTH)
    predicted = kerasmodel.predict(something)
    predicted = np.array(predicted[0])


    best_three = predicted.argsort()[-3:][::-1]

    best_parties = list()
    for number in best_three:
      best_parties.append((party_names[number].split('.')[0],int(100*predicted[number])))
    toreturn = dict()
    toreturn["understood"] = sentence_understood
    toreturn["best_parties"] = best_parties
    return json.dumps(toreturn)


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,predicttext')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

if __name__ == "__main__":
    path = os.path.join(SAVED_TRAINDATA_DIR,FILENAME_SAVED_DATA)
    if os.path.exists(path):
        loaded = pickle.load(open(path, "rb"))
        X_test = loaded['X_test']
        y_test = loaded['y_test']
        id_of_word_getter = loaded["idofwordgetter"]
        party_names = loaded["partynames"]
    else:

        parties_and_sentences= partyprogram_loader.get_parties_and_sentences(PARTIJPATH)
        vocab_list = partyprogram_loader.vocab_from_sentences(parties_and_sentences)
        id_of_word_getter = partyprogram_loader.IdOfWordGetter(vocab_list)
        party_count = len(parties_and_sentences.keys())
        party_names = list(parties_and_sentences.keys())
        word_count = len(vocab_list)

        (X_train, y_train), (X_test, y_test) = partyprogram_loader.load_data(parties_and_sentences, id_of_word_getter)

        X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
        X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)
        y_train = np_utils.to_categorical(y_train, party_count)
        prevytest = y_test
        y_test = np_utils.to_categorical(y_test, party_count)
        tosave = dict()
        tosave['y_test'] = y_test
        tosave['X_test'] = X_test
        tosave["idofwordgetter"] = id_of_word_getter
        tosave["partynames"] = party_names
        path = os.path.join(SAVED_TRAINDATA_DIR,FILENAME_SAVED_DATA)
        pickle.dump(tosave, open(path, "wb"))

    path = os.path.join(SAVED_TRAINDATA_DIR, FILENAME_SAVED_MODEL)
    if os.path.exists(path):
        kerasmodel = model.load_model(path)
    else:
        kerasmodel = model.get_model(word_count,EMBEDDING_VECTOR_LENGTH,MAX_REVIEW_LENGTH,party_count)
        model.train_model(kerasmodel, X_train, y_train)
        kerasmodel.save(path)


    print("Done training and loading model")
    # Final evaluation of the model
    scores = kerasmodel.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("commit test")

    app.run(host="0.0.0.0",port=5009, ssl_context=context)
