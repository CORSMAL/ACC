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
import glob
import librosa
import pandas as pd
from config import SAMPL_RATE, DEMO_SAMPLES

'''
Functions from data_parser.py, from the Official CORSMAL Container Manipulation Evaluation
toolkit, by Alessio Xompero (corsmal-challenge@qmul.ac.uk).
'''

fnames_filling  = ['fi0_fu0', 'fi1_fu1', 'fi1_fu2', 'fi2_fu1', 'fi2_fu2', 'fi3_fu1', 'fi3_fu2']
fnames_filling2 = ['fi0_fu0', 'fi1_fu1', 'fi1_fu2', 'fi2_fu1', 'fi2_fu2']

# Generate waveform and spectrogram from a file's name
def gen_wave_spectro(containerpath, f):

    audiofile = containerpath + '/audio/' + f + '_audio.wav'
    audio, _ = librosa.load(audiofile, res_type='kaiser_fast')
    X = librosa.stft(audio)
    spectro = librosa.amplitude_to_db(abs(X))

    return audio, spectro


def parse_label(name):

    fi = int(name[5])
    fu = int(name[9])

    if (fi == 0 or fu == 0): cl = 0
    elif (fi == 1 and fu == 1): cl = 1  # Pasta both fillings
    elif (fi == 1 and fu == 2): cl = 2
    elif (fi == 2 and fu == 1): cl = 3  # Rice both fillings
    elif (fi == 2 and fu == 2): cl = 4
    elif (fi == 3 and fu == 1): cl = 5  # Water both fillings
    elif (fi == 3 and fu == 2): cl = 6
    else: print('Wrong class assignment')

    return cl


def populate_filenames(mode):
	list_filenames = []
	for s in range(0, 3):
		str_s = 's{:d}_'.format(s)

		for b in range(0, 2):
			str_b = '_b{:d}_'.format(b)

			for l in range(0, 2):
				str_l = 'l{:d}'.format(l)

				if mode == 0:
					for f in fnames_filling:
						list_filenames.append(str_s + f + str_b + str_l)
				else:
					for f in fnames_filling2:
						list_filenames.append(str_s + f + str_b + str_l)

	return list_filenames


def TrainingDataParser(datapath):
    data = []
    for objid in range(1, 10):
        containerpath = datapath + '/{:d}'.format(objid)
        sequence = 0

        if objid < 7:
            list_files = populate_filenames(0)
        else:
            list_files = populate_filenames(1)

        for f in list_files:
            # File loading
            audio, spec = gen_wave_spectro(containerpath, f)
            label = parse_label(f)
            data.append([f, int('{:d}'.format(objid)), sequence, audio, spec, label])
            sequence += 1

    return pd.DataFrame(data, columns=['file_name', 'container', 'sequence', 'waveform', 'spectrogram', 'label'])


def PublicTestingDataParser(datapath):
    data = []
    for objid in range(10, 13):
        containerpath = datapath + '/{:d}'.format(objid)
        sequence = 0

        list_files = []
        if objid < 12:
            for j in range(0, 84):
                list_files.append('{:04d}'.format(j))
        else:
            for j in range(0, 60):
                list_files.append('{:04d}'.format(j))

        for f in list_files:
            # File loading
            audio, spec = gen_wave_spectro(containerpath, f)
            data.append([f, int('{:d}'.format(objid)), sequence, audio, spec])
            sequence += 1

    return pd.DataFrame(data, columns=['file_name', 'container', 'sequence', 'waveform', 'spectrogram'])


def PrivateTestingDataParser(datapath):
    data = []
    for objid in range(13, 16):
        containerpath = datapath + '/{:d}'.format(objid)
        sequence = 0

        list_files = []
        if objid < 15:
            for j in range(0, 84):
                list_files.append('{:04d}'.format(j))
        else:
            for j in range(0, 60):
                list_files.append('{:04d}'.format(j))

        for f in list_files:
            # File loading
            audio, spec = gen_wave_spectro(containerpath, f)
            data.append([f, int('{:d}'.format(objid)), sequence, audio, spec])
            sequence += 1

    return pd.DataFrame(data, columns=['file_name', 'container', 'sequence','waveform', 'spectrogram'])

def load_new_containers(path):
    
    other_data = []
    sequence = 0
    
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith('.wav'):

                audio, _ = librosa.load(filepath, res_type='kaiser_fast') # kaiser_best kaiser_fast
                # spec = extract_spectro(audio)
                X = librosa.stft(audio)
                spec = librosa.amplitude_to_db(abs(X))
                
                f_name = os.path.split(filepath)[1]

                label = int(f_name[-5])
                container = int(f_name[1:3])
                
                other_data.append([os.path.split(filepath)[1], container, sequence, audio, spec, label])
                sequence += 1
                
    other_df = pd.DataFrame(other_data, columns=['file_name', 'container', 'sequence', 'waveform', 'spectrogram', 'label'])
    print('Finished feature extraction from', len(other_df), 'files')
                
    return other_df


def load_demo_samples(path):

    data = []

    for filename in DEMO_SAMPLES:
        filepath = path + os.sep + filename

        audio, _ = librosa.load(filepath, res_type='kaiser_fast') # kaiser_best kaiser_fast
        X = librosa.stft(audio)
        spec = librosa.amplitude_to_db(abs(X))
        
        file_name = os.path.basename(filepath)
        label = parse_label(file_name)
        data.append([file_name, audio, spec, label])

        # print('Label:', label)
            
    df = pd.DataFrame(data, columns=['file_name','waveform', 'spectrogram', 'label'])

    return df

def load_dataset(path, data_split):

    if data_split == 'train' or data_split == 'train_dev':
        print('Loading training dataset...')
        df = TrainingDataParser(path)

    elif data_split == 'test' or data_split == 'test_dev':
        print('Loading public test dataset...')
        df = PublicTestingDataParser(path)

    elif data_split == 'private_test':
        # Replace <private_test> with real name
        print('Loading private test dataset...')
        df = PrivateTestingDataParser(path)

    elif data_split == 'new_containers':
        print('Loading new containers dataset...')
        df = load_new_containers(path)

    elif data_split == 'demo':
        print('Loading demo samples from dataset')
        try:
            df = load_demo_samples(path)
            print('OK\n')
        except Exception as e:
            print('ERROR: Unable to load dataset, please check argument --datapath \n')
            print(e)
        
    else:
        print('\nDataset not found. Check datapath.\n')

    return df