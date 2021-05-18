#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing the environment and setup and demo of ACC.
"""
# -----------------------------------------------------------------------------
# Authors: 
#  	- Santiago Donaher
# 	- Alessio Xompero: a.xompero@qmul.ac.uk
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
'''
Arguments:
Arg datapath: Path to dataset. (e.g. CORSMAL Container Manipulation dataset).
    Path as can be relative or absolute.
    E.g. > 'D:/CORSMAL_Dataset/'

'''

from scripts.data_loading import load_dataset
from methods.processing.audio_processing import audio_processing
from methods.processing.data_processing import prepare_demo_data
from scripts.model_build import pretrained_model
from scripts.model_run import model_estimate

import pandas as pd
import argparse

from config import DATASET_PATH, CAT2FIL, DEMO_SAMPLES

# RUNTIME PROCEDURE
if __name__ == '__main__':

	# Arguments
	# parser = argparse.ArgumentParser(description='ACC Network')
	# parser.add_argument('--datapath', default=DATASET_PATH, type=str)
	# args = parser.parse_args()

	# data_path = args.datapath

	data_path = 'CCM_samples/demo'

	print('\nRunning demo')
	print('Data path:\t{}\n'.format(data_path))
	
	# Load data
	df = load_dataset(data_path, 'demo')

	# Audio Processing
	df = audio_processing(df)
	print('OK\n')

	x_val, y_val = prepare_demo_data(df)
	
	print('\nLoading ACC')
	try:
		model = pretrained_model('acc')
		print('OK\n')
	except Exception as e:
		print('ERROR: Unable to load models, please check if all .json and .h5 files are in /acc-net/methods/acc/models/\n')
		print(e)

	print('Running ACC')
	try:
		y_pred, _ = model_estimate(model, x_val)
	except Exception as e:
		print('ERROR: Unable to run model, please check hardware setup (if using GPU) and TensorFlow version\n')
		print(e)
	
	print('\nSAMPLE\t\t\t\tLABEL\t\tPREDICTION')
	for (sample, i_pred, i_val) in zip(DEMO_SAMPLES, y_pred, y_val.tolist()):
		print('{}\t{}   \t{}'.format(sample, CAT2FIL[i_val], CAT2FIL[i_pred]))

	if y_pred == y_val.tolist(): print('\nOK: Ready to run main.py')

	print('\nend_of_script')
