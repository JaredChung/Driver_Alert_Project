from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from keras import backend as K
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, RMSprop


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import theano






def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    data = pd.read_csv("Train-IsAlert-secondRound.csv")
    X = data.drop(['TrialID','IsAlert'], axis=1)
    Y = data.IsAlert
    train_X, test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3, random_state=42)
    

    train_X = train_X.values.astype(theano.config.floatX)
    test_X = test_X.values.astype(theano.config.floatX)
    
    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)


    train_Y = np_utils.to_categorical(train_Y.values)
    test_Y = np_utils.to_categorical(test_Y.values)
    return train_X, train_Y, test_X, test_Y


def model(train_X, train_Y, test_X, test_Y):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    
    model = Sequential()
    model.add(Dense(500,input_shape=(train_X.shape[1],)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense({{choice([512, 1024])}}))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense({{choice([512, 1024])}}))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))
    

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(500))
        # We can also choose between complete sets of layers
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(train_Y.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(train_X, train_Y,
              batch_size={{choice([128, 256])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(test_X, test_Y))
    score, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials())
    data = pd.read_csv("Train-IsAlert-secondRound.csv")
    X = data.drop(['TrialID','IsAlert'], axis=1)
    Y = data.IsAlert
    train_X, test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3, random_state=42)


    train_X = train_X.values.astype(theano.config.floatX)
    test_X = test_X.values.astype(theano.config.floatX)

    scaler = StandardScaler()
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    train_Y = np_utils.to_categorical(train_Y.values)
    test_Y = np_utils.to_categorical(test_Y.values)
    print ("Print Best Model")
    print (best_model.summary())
    print ("Configuration")
    print (model.get_config())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(test_X, test_Y))