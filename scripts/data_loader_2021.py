# -----------------------------------------------------------------------------
# Authors:
#   - Santiago Donaher: s.donaher@qmul.ac.uk
#   - Alessio Xompero: a.xompero@qmul.ac.uk
#
#   Edited Date: 2021/12/20
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
import argparse
import sys
import json
import pandas as pd
import librosa
# import os
# import glob

from config import SAMPL_RATE, DEMO_SAMPLES, NDATASET_PATH, NANNOTAT_PATH_JSON, NANNOTAT_PATH_CSV

def search_annot_json(id, annot_file):

    for i in annot_file['annotations']:
        if i['id'] == id:
            cont_id = i['container id']
            f_type  = i['filling type']
            f_level = i['filling level']
            # print('scenario', i['scenario'])
            # print('Label (t, l):', f_type, f_level)
            break

    return cont_id, f_type, f_level

def parse_label(fi, fu):

    if (fi == 0 or fu == 0): cl = 0
    elif (fi == 1 and fu == 1): cl = 1  # Pasta both fillings
    elif (fi == 1 and fu == 2): cl = 2
    elif (fi == 2 and fu == 1): cl = 3  # Rice both fillings
    elif (fi == 2 and fu == 2): cl = 4
    elif (fi == 3 and fu == 1): cl = 5  # Water both fillings
    elif (fi == 3 and fu == 2): cl = 6
    else: print('Wrong class assignment')

    return cl

def gen_wave_spectro(audiofile):

    audio, _ = librosa.load(audiofile, res_type='kaiser_fast')
    X = librosa.stft(audio)
    spectro = librosa.amplitude_to_db(abs(X))

    return audio, spectro

def ParseFile(containerpath, f):

	## AUDIO DATA
    audiofile = containerpath + 'audio/' + f + '.wav'
    return audiofile

	## RGB DATA
	# for cam_id in range(1,3):
	# 	rgbvideo = containerpath + 'view{:d}/rgb/'.format(cam_id) + f + '.mp4'


#### TRAIN SET ###
def TrainSetDataParser(datapath, annot_file):
    data = []
    for j in range(0,684):
        audiofile_p = ParseFile(str(datapath + 'train/'), '{:06d}'.format(j))
        # print('\nReading file: ', audiofile_p)
        audio, spec = gen_wave_spectro(audiofile_p)
        # print(audio[:10])

        cont, f_type, f_level = search_annot_json(int(j), annot_file)
        label = parse_label(f_type, f_level)
        # print('Label', label)

        # data.append([f, int('{:d}'.format(objid)), sequence, audio, spec, label])
        data.append([j, cont, audio, spec, label])

    return pd.DataFrame(data, columns=['id', 'container', 'waveform', 'spectrogram', 'label'])
    # return pd.DataFrame(data, columns=['file_name', 'container', 'sequence', 'waveform', 'spectrogram', 'label'])

#### PUBLIC TEST SET ###
def PublicTestSetDataParser(datapath):
    data = []
    for j in range(0,228):

        audiofile_p = ParseFile(str(datapath + 'test_pub/'), '{:06d}'.format(j))
        audio, spec = gen_wave_spectro(audiofile_p)

        data.append([j, audio, spec])

    return pd.DataFrame(data, columns=['id', 'waveform', 'spectrogram'])


#### PRIVATE TESTING SET ###
def PrivateTestSetDataParser(datapath):
    data = []
    for j in range(0,228):

        audiofile_p = ParseFile(str(datapath + 'test_pri/'), '{:06d}'.format(j))
        audio, spec = gen_wave_spectro(audiofile_p)

        data.append([j, audio, spec])

    return pd.DataFrame(data, columns=['id', 'waveform', 'spectrogram'])

# ---
def load_dataset_2021(path, data_split):

    with open(NANNOTAT_PATH_JSON, "r") as read_file:
        annot_json = json.load(read_file)

    if data_split == 'train':
        print('Loading training dataset...')
        df = TrainSetDataParser(NDATASET_PATH, annot_json)

    elif data_split == 'test':
        print('Loading public test dataset...')
        df = PublicTestSetDataParser(NDATASET_PATH)





    return df

# Just to test loader
if __name__ == '__main__':

    print('\nChecking version')
    print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))

    # Parse arguments
    parser = argparse.ArgumentParser(description='2021 CCM data loader')
    parser.add_argument('--datapath', default=NDATASET_PATH, type=str)
    parser.add_argument('--set', default='train', type=str, choices=['train','pubtest','privtest'])
    args = parser.parse_args()

    print('Loading {} data split(s) from {}'.format(args.set, args.datapath))

    print('JSON file found:', os.path.exists(NANNOTAT_PATH_JSON))
    with open(NANNOTAT_PATH_JSON, "r") as read_file:
        annot_json = json.load(read_file)

    # LOAD DATA
    if args.set == 'train':
        TrainSetDataParser(args, annot_json)
    elif args.set == 'pubtest':
        PublicTestSetDataParser(args)
    elif args.set == 'privtest':
        PrivateTestSetDataParser(args)

    # LOAD ANNOTATIONS (INTEGRATE INTO FUNCION ABOVE)
    # from csv
    # print('CSV file found:', os.path.exists(NANNOTAT_PATH_CSV))
    # annot_csv = pd.read_csv(NANNOTAT_PATH_CSV, sep=',')
    # print(annot_csv.head())



