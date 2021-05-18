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

from methods.acc.src.acc_components import acc_action, acc_pouring, acc_shaking

import os
from tensorflow.keras import Model
from tensorflow.keras.models import model_from_json

import pickle

from config import IMG_SIZE, TR_ACC

def raw_model(model_use, y_train):

    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    n_class = y_train.shape[1]

    if model_use == 'acc_action':
        model = acc_action(input_shape, n_class)

    elif model_use == 'acc_pouring':
        model = acc_pouring(input_shape, n_class)

    elif model_use == 'acc_shaking':
        model = acc_shaking(input_shape, n_class)

    else: print('Wrong model selected.')

    print('\nInitialized model:', model_use)
    # print(model.summary())

    return model

def pretrained_model(model_use):

    current_path = os.path.abspath(os.path.dirname(__file__))

    if model_use == 'acc':
        
        dir_models = current_path + '/../methods/acc/models/' + TR_ACC

        model_act_json_p = dir_models + '_action.json'
        model_act_weig_p = dir_models + '_action.h5'

        model_pour_json_p = dir_models + '_pouring.json'
        model_pour_weig_p = dir_models + '_pouring.h5'

        model_shak_json_p = dir_models + '_shaking.json'
        model_shak_weig_p = dir_models + '_shaking.h5'
    
        # Action Model
        json_file = open(model_act_json_p, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_action = model_from_json(loaded_model_json)
        model_action.load_weights(model_act_weig_p)
        model_action.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

        # Pouring Model
        json_file = open(model_pour_json_p, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_pouring = model_from_json(loaded_model_json)
        model_pouring.load_weights(model_pour_weig_p)
        model_pouring.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

        # Shaking Model
        json_file = open(model_shak_json_p, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_shaking = model_from_json(loaded_model_json)
        model_shaking.load_weights(model_shak_weig_p)
        model_shaking.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

        model = [model_action, model_pouring, model_shaking]

        print('{} loaded succesfully'.format(model_use))

    else: print('Wrong model selected.')

    return model

