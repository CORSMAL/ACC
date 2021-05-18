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

import librosa.util
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

from config import SAMPL_RATE

def plot_file(file):

    plt.subplot(2, 1, 1)
    librosa.display.waveplot(file.waveform, sr=SAMPL_RATE)
    plt.title('File Name: {}. Class: {}'.format(file.file_name, file.label))
    plt.subplot(2, 1, 2)
    librosa.display.specshow(file.spectrogram, sr=SAMPL_RATE, x_axis='time', y_axis='hz')
    plt.show()

def plot_rand_files(df):

    row = 3
    col = 3
    rand_aud = []

    # Random audios
    for n in range(row*col):
        rand_aud.append(rd.randint(0, df.shape[0]-1))

    # Waveforms
    axes=[]
    fig=plt.figure(figsize=(12, 7))
    for a in range(row*col):
        n = rand_aud[a]
        wave = df.waveform[n]
        axes.append(fig.add_subplot(row, col, a+1))
        # subplot_title=(cat2fil[df.label[n]])
        # axes[-1].set_title(subplot_title)
        librosa.display.waveplot(wave, sr=SAMPL_RATE)
    fig.tight_layout()
    plt.show(block = False)

    # Spectrograms
    axes=[]
    fig=plt.figure(figsize=(12, 7))
    for a in range(row*col):
        n = rand_aud[a]
        spec = df.spectrogram[n]
        axes.append(fig.add_subplot(row, col, a+1))
        # subplot_title=(cat2fil[df.label[n]])
        # axes[-1].set_title(subplot_title)
        librosa.display.specshow(spec, sr=SAMPL_RATE, x_axis='time', y_axis='hz')
    fig.tight_layout()
    plt.show()