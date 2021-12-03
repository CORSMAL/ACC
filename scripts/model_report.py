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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import tikzplotlib
import pickle

from scripts.results_parser import estimations2csv

from config import BATCH_SIZE

def confusion_matrix_data(model_use, y_test, y_pred):

    if model_use == 'acc_action':
        target_names = ['Empty','Pouring','Shaking']

    elif model_use == 'acc_pouring':
        target_names = ['pasta_50','pasta_90','rice_50','rice_90','water_50','water_90']

    elif model_use == 'acc_shaking':
        target_names = ['pasta_50','pasta_90','rice_50','rice_90']

    else:
        target_names = ['Empty','pasta_50','pasta_90','rice_50','rice_90','water_50','water_90']

    return target_names

def plot_confusion_matrix(cm, path_name, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path_name)


# Function to save training curves AND the trained model itself
def train_report(model_use, model, history, x_test, y_test, outdir):

    # Check if dir exist and if not, create
    path = os.path.join(outdir, 'train')
    if not os.path.isdir(path): os.makedirs(path)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    y_pred  = model.predict(x_test)

    # Create new dir
    id_now = datetime.now().strftime("%m%d-%H%M%S")
    loss = str("{:.3f}".format(results[0]))
    accu = str("{:.4f}".format(results[1]))
    dir_name = id_now +'_'+ model_use +'_'+ accu +'_'+ loss
    path = os.path.join(path, dir_name)
    os.makedirs(path)

    # All file names
    model_n      = os.path.join(path, model_use+'.json')
    weights_n    = os.path.join(path, model_use+'_w.h5')
    accu_curve_n = os.path.join(path, model_use+'_accu_curve')
    loss_curve_n = os.path.join(path, model_use+'_loss_curve')
    confusion_n  = os.path.join(path, model_use+'_CM.png')

    # Save model
    model_json = model.to_json()
    with open(model_n, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_n)

    # Save training curves
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(accu_curve_n+'.png')
    # tikzplotlib.save(accu_curve_n+'.tex')
    # plt.show()
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(loss_curve_n+'.png')
    # tikzplotlib.save(loss_curve_n+'.tex')
    # plt.show()
    plt.clf()

    print('\nFinal results with test data:')
    print('Accuracy\t{}\nLoss\t\t{}'.format(accu, loss))

    # Calculate and save Confusion Matrix and metrics (Precission, Recall, etc)
    target_names = confusion_matrix_data(model_use, y_test, y_pred)
    
    try:
        matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        print('\nClassification report:\n', classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names))
        
        print('Confusion Matrix:\nLabels order {}\n{}'.format(target_names, matrix))
        plot_confusion_matrix(cm=matrix, path_name=confusion_n, classes=target_names, title='Confusion Matrix')
    except:
        print('Error training model, check exported curves')

    print('\nModel, curves and CM saved as {}'.format(dir_name))

def test_report(model_use, dataset, df, y_pred, action_pred, outdir):

    # Check if dir exist and if not, create
    path = os.path.join(outdir, 'test')
    if not os.path.isdir(path): os.makedirs(path)

    # Create new dir
    id_now = datetime.now().strftime('%m%d-%H%M%S')
    dir_name = id_now +'_'+ model_use +'_'+ dataset
    path = os.path.join(path, dir_name)
    os.makedirs(path)

    # Estimations to CSV
    df_results = estimations2csv(y_pred, df.container.to_list(), df.sequence.to_list(), action_pred)
    expt_path = os.path.join(path, 'estimations.csv')
    df_results.to_csv(expt_path, index=False)
    abs_path = os.path.abspath(expt_path)
    print('\nEstimations exported to:\n', abs_path)

# def test_report_2021(model_use, dataset, df, y_pred, action_pred, outdir):

    # Check if dir exist and if not, create
    # path = os.path.join(outdir, 'test')
    # if not os.path.isdir(path): os.makedirs(path)

    # # Create new dir
    # id_now = datetime.now().strftime('%m%d-%H%M%S')
    # dir_name = id_now +'_'+ model_use +'_'+ dataset
    # path = os.path.join(path, dir_name)
    # os.makedirs(path)

    # # Estimations to CSV
    # df_results = estimations2csv(y_pred, df.container.to_list(), df.sequence.to_list(), action_pred)
    # expt_path = os.path.join(path, 'estimations.csv')
    # df_results.to_csv(expt_path, index=False)
    # abs_path = os.path.abspath(expt_path)
    # print('\nEstimations exported to:\n', abs_path)
    

    
    

    

