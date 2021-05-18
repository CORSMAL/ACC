# -----------------------------------------------------------------------------
# Authors: 
#   - Santiago Donaher
#   - Alessio Xompero: a.xompero@qmul.ac.uk
#
# MIT License

# Copyright (c) 2021 CORSMAL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

import tensorflow as tf

from config import LEARNING_RATE, IMG_SIZE

def acc_cascade(model, x_val):

    model_action  = model[0]
    model_pouring = model[1]
    model_shaking = model[2]

    action_pred = model_action.predict(x_val, verbose=0)
    action_pred = action_pred.argmax(axis=1)

    y_pred = []

    for n in range(x_val.shape[0]):

        tmp = x_val[n].reshape(1, IMG_SIZE, IMG_SIZE, 1)

        if action_pred[n] == 0:
            y_pred.append(int(0))

        elif action_pred[n] == 1:
            pred = model_pouring.predict(tmp, verbose=0)
            y_pred.append(np.argmax(pred)+1)

        elif action_pred[n] == 2:
            pred = model_shaking.predict(tmp, verbose=0)
            y_pred.append(np.argmax(pred)+1)


        else:
            print('Error with action predictor')

    return y_pred, action_pred

# Models from experiments
def acc_action(input_shape, n_class):

    inp = Input(shape=input_shape)

    x = Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(3)(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(3)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name="action_model")

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

    return model

def acc_pouring(input_shape, n_class):

    dr_rate = 0.3

    inp = Input(shape=input_shape)

    x = Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Dropout(dr_rate)(x)
    x = MaxPooling2D(3)(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Dropout(dr_rate)(x)
    x = MaxPooling2D(3)(x)

    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Dropout(dr_rate)(x)
    x = MaxPooling2D(3)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dr_rate)(x)
    out = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name='pouring_model')

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

    return model

def acc_shaking(input_shape, n_class):

    inp = Input(shape=input_shape)

    x = Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(3)(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(3)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out, name='shaking_model')

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=['categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])

    return model


    