from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import load_model

import numpy as np

def get_model(words_top_count,embedding_vector_length,max_review_length,party_count):
    model = Sequential()
    model.add(Embedding(words_top_count, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(party_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def train_model(model,X_train,y_train,epochs=3,batch_size=64):
    model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=1)

def load_model(filename):
    model = load_model(filename)
    return model


