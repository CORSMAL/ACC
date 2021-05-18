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

import librosa
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from skimage.transform import resize

from config import IMG_SIZE, SAMPL_RATE, AUDIO_LEN
# from config import LOW_THRESH, MID_THRESH, HIGH_THRESH, LOUD_AUDIO, SOFT_AUDIO

import warnings
warnings.filterwarnings('ignore')

def onset_detection(df):

    sr = SAMPL_RATE
    for n in range(df.shape[0]):

        file = df.waveform[n]
        onset = librosa.onset.onset_detect(y=file, sr=sr, units='samples')
        try:
            sec_onset = onset[1]
            df.waveform[n] = file[sec_onset:]
        except IndexError:
            df.waveform[n] = file

    return df

def normalize_len(df):

    f_len = SAMPL_RATE * AUDIO_LEN

    df['waveform'] = [file[:f_len] for file in df.waveform]
    # In case some file was shorter, the remaining empty part is zero-padded
    df['waveform'] = [np.pad(col, (0, f_len-len(col)), 'constant') for col in df.waveform if len(df.waveform) < f_len]

    return df

def update_spectro(df):

    df = df.drop(['spectrogram'], axis=1)

    new_spect = []
    for s in range(df.shape[0]):
        X = librosa.stft(df['waveform'][s])
        new_spect.append(librosa.amplitude_to_db(abs(X)))

    spect_values = pd.Series(new_spect)
    df.insert(loc=0, column='spectrogram', value=spect_values)

    return df

def spectro_resize(spect, f_x, f_y):
    return resize(spect, (f_x, f_y))



def audio_processing(df):

    print('Processing data:')
    print('1. Normalizing amplitudes')
    df['waveform'] = [librosa.util.normalize(file) for file in df.waveform]

    print('2. Finding onsets')
    df = onset_detection(df)

    print('3. Normalizing lengths')
    df = normalize_len(df)

    print('4. Updating spectrograms')
    df = update_spectro(df)

    print('5. Resizing spectrograms')
    df['spectrogram'] = [spectro_resize(i, IMG_SIZE, IMG_SIZE) for i in df['spectrogram']]

    return df
