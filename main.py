
import os
import pickle

from keras.preprocessing import sequence
from keras.utils import np_utils
from flask import Flask
from flask import request


app = Flask(__name__)

import model
import partyprogram_loader

# GLOBALS YOU CAN SET AND PLAY WITH
FILENAME_SAVED_MODEL = 'party_model.h5'
FILENAME_SAVED_DATA = 'data.p'

PARTIJPATH = 'partijprogrammas'
EMBEDDING_VECTOR_LENGTH = 32
MAX_REVIEW_LENGTH = 20

kerasmodel = None

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict')
def predict_party():
    sentence = request.headers.get('predicttext')
    print(request.headers)

    print("Got sentence: " + sentence)
    return "Thank you"

if __name__ == "__main__":
    if os.path.exists(FILENAME_SAVED_DATA):
        loaded = pickle.load(open(FILENAME_SAVED_DATA, "rb"))
        X_test = loaded['X_test']
        y_test = loaded['y_test']
    else:

        parties_and_sentences= partyprogram_loader.get_parties_and_sentences(PARTIJPATH)
        vocab_list = partyprogram_loader.vocab_from_sentences(parties_and_sentences)
        id_of_word_getter = partyprogram_loader.IdOfWordGetter(vocab_list)
        party_count = len(parties_and_sentences.keys())
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
        pickle.dump(tosave, open(FILENAME_SAVED_DATA, "wb"))

    if os.path.exists(FILENAME_SAVED_MODEL):
        kerasmodel = model.load_model(FILENAME_SAVED_MODEL)
    else:
        kerasmodel = model.get_model(word_count,EMBEDDING_VECTOR_LENGTH,MAX_REVIEW_LENGTH,party_count)
        model.train_model(kerasmodel, X_train, y_train)
        kerasmodel.save(FILENAME_SAVED_MODEL)


    print("Done training and loading model")
    # Final evaluation of the model
    scores = kerasmodel.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("commit test")

    app.run(host="0.0.0.0",port=5000)