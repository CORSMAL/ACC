# -----------------------------------------------------------------------------
# Authors: 
#   - Santiago Donaher: s.donaher@qmul.ac.uk
#   - Alessio Xompero: a.xompero@qmul.ac.uk
#
#   Edited Date: 2021/11/30
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
# import os
# import glob
# import librosa
# from config import SAMPL_RATE, DEMO_SAMPLES

# New globals
NDATASET_PATH = 'D:/CCM_dataset/CCM_2021/'
NANNOTAT_PATH_json = 'D:/CCM_dataset/CCM_2021/ccm_train_annotation.json'
NANNOTAT_PATH_csv = 'D:/CCM_dataset/CCM_2021/ccm_train_annotation.csv'

def search_annot_json(id, annot_file):

    for i in annot_file['annotations']:
        if i['id'] == id:
            cont_id = i['container id']
            # print('scenario', i['scenario'])
            f_type  = i['filling type']
            f_level = i['filling level']            
            break

    return 1

def ParseFile(containerpath, f):
	## RGB DATA
	for cam_id in range(1,3):
		rgbvideo = containerpath + 'view{:d}/rgb/'.format(cam_id) + f + '.mp4'
		# print(rgbvideo)

	## AUDIO DATA
	audiofile = containerpath + 'audio/' + f + '.wav'
	# print(audiofile)

#### TRAIN SET ###
def TrainSetDataParser(args, annot_file):
	for j in range(0,684):
		ParseFile(str(args.datapath + 'train/'), '{:06d}'.format(j))

        search_annot_json(int(j), annot_file)

		# ParseFile(args.datapath, '{:06d}'.format(j))

#### PUBLIC TEST SET ###
def PublicTestSetDataParser(args):
	for j in range(0,228):
		ParseFile(args.datapath, '{:06d}'.format(j))

#### PRIVATE TESTING SET ###
def PrivateTestSetDataParser(args):
	for j in range(0,228):
		ParseFile(args.datapath, '{:06d}'.format(j))

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

    print('JSON file found:', os.path.exists(NANNOTAT_PATH_json))
    with open(NANNOTAT_PATH_json, "r") as read_file:
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
    # print('CSV file found:', os.path.exists(NANNOTAT_PATH_csv))
    # annot_csv = pd.read_csv(NANNOTAT_PATH_csv, sep=',')
    # print(annot_csv.head())




