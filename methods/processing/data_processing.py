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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from config import IMG_SIZE

def select_train_data(df, model_use):

    if model_use == 'acc_action':
        df.rename(columns = {'label':'temp_label'}, inplace = True)
        df.loc[df['container']  <  7, 'label'] = np.int(1)
        df.loc[df['container']  >= 7, 'label'] = np.int(2)
        df.loc[df['temp_label'] == 0, 'label'] = np.int(0)
        df = df.drop(['temp_label'], axis=1)

    elif model_use == 'acc_pouring':
        df = df.loc[(df['container'] != 7) & (df['container'] != 8) & (df['container'] != 9)]
        df = df.loc[(df['label'] != 0)]

    elif model_use == 'acc_shaking':
        temp_df = df.loc[(df['container'] == 7) | (df['container'] == 8) | (df['container'] == 9)]
        temp_df = temp_df.loc[(temp_df['label'] != 0)]
        temp_df = temp_df.reset_index()
        df = temp_df.drop(['index'], axis=1)

    return df

def data_shapes(x_train, y_train, x_test, y_test):

    print('\nData shape:')
    print("Train Data:", x_train.shape)
    print("Train Labels:", y_train.shape)
    print("Test Data:", x_test.shape)
    print("Test Labels:", y_test.shape)
    print('\n')

def prepare_train_data(df, model_use):

    df = select_train_data(df, model_use)

    X = np.array(df.spectrogram.tolist())
    y = np.array(df.label.tolist())
    lec = LabelEncoder()
    yy = to_categorical(lec.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42, stratify=yy)

    x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
    data_shapes(x_train, y_train, x_test, y_test)
        
    return x_train, x_test, y_train, y_test

def prepare_test_data(df, dataset):
    
    x_val = np.array(df.spectrogram.tolist())

    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    print('Test data shape:', x_val.shape)

    return x_val

def prepare_demo_data(df):
    
    x_val = np.array(df.spectrogram.tolist())
    y = np.array(df.label.tolist())

    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    print('Demo data shape:', x_val.shape)

    return x_val, y