#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for audio pouring/shaking estimation.
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
ARGUMENT OPTIONS

Arg datapath: Path to dataset. (e.g. CORSMAL Container Manipulation dataset).
	For training, labels are required.
    Path as can be relative or absolute.
    E.g. > 'D:/CORSMAL_Dataset/'

Arg outdir: Path to output directory. Will export trained models and curves if training,
	or predictions as CSV if testing
    Path as can be relative or absolute.
    E.g. > 'D:/acc/outputs/'

Arg mode: Running option. Training or testing ACC in a dataset.
    > 'train' / 'test'

Arg data_split: To select which data split (or other dataset such as new container) to use.
	Default: test. If mode is in train, this argument will automatically be set to train
	> 'train','test','private_test','new_containers'

Arg model: Select the model to train or test. Available models:
    ACC Network:
        + To train: either whole ACC: 'acc', or each sub-model individually: > 'acc_<action/pouring/shaking>'
        + To test: > 'acc'

    '''

from scripts.data_loading import load_dataset
from scripts.data_loader_2021 import load_dataset_2021
from methods.processing.audio_processing import audio_processing
from methods.processing.data_processing import prepare_train_data, prepare_test_data
from scripts.model_build import raw_model, pretrained_model
from scripts.model_run import model_train, model_estimate
from scripts.model_report import train_report, test_report
from scripts.data_visualization import plot_rand_files

import pandas as pd
import argparse

from config import DATASET_PATH, OUTPUT_PATH

# RUNTIME PROCEDURE
if __name__ == '__main__':

	# Arguments
	parser = argparse.ArgumentParser(description='ACC Network')

	parser.add_argument('--datapath', default=DATASET_PATH, type=str)
	parser.add_argument('--outdir', default=OUTPUT_PATH, type=str)
	parser.add_argument('--version', default=2020, type=int, choices=[2020, 2021])
	parser.add_argument('--mode', default='test', type=str, choices=['train','test'])
	parser.add_argument('--data_split', default='test', type=str, choices=['train','test','private_test','new_containers'])
	parser.add_argument('--model', default='acc', type=str, choices=['acc','acc_action','acc_pouring','acc_shaking'])
	args = parser.parse_args()

	mode      = args.mode
	model_use = args.model
	dataset   = args.data_split
	data_path = args.datapath
	out_path  = args.outdir
	y_version = args.version

	if mode == 'train':
		dataset = 'train'
	data_path += dataset

	print('\n{}ing {} on {} dataset'.format(mode, model_use, dataset))
	print('Data path:\t{}\nOutput path:\t{}\n'.format(data_path, out_path))
	
	# Load data
	if y_version == 2021:
		df = load_dataset_2021(data_path, dataset)
	else: 
		df = load_dataset(data_path, dataset)

	# Audio Processing
	df = audio_processing(df)
	# plot_rand_files(df)	# Uncoment for samples visualization

	# Train raw model
	if mode == 'train' and model_use != 'acc':
		x_train, x_test, y_train, y_test = prepare_train_data(df, model_use)
		model = raw_model(model_use, y_train)
		model, history = model_train(model, x_train, x_test, y_train, y_test)
		train_report(model_use, model, history, x_test, y_test, args.outdir)

	if mode == 'train' and model_use == 'acc':
		for acc_comp in ['acc_action', 'acc_pouring', 'acc_shaking']:
			x_train, x_test, y_train, y_test = prepare_train_data(df, acc_comp)
			model = raw_model(acc_comp, y_train)
			model, history = model_train(model, x_train, x_test, y_train, y_test)
			train_report(acc_comp, model, history, x_test, y_test, args.outdir)

	# Test with pre-trained model
	if mode == 'test':
		x_val = prepare_test_data(df, dataset)
		model = pretrained_model(model_use)
		y_pred, action_pred = model_estimate(model, x_val)
		test_report(model_use, dataset, df, y_pred, action_pred, args.outdir)

	print('\nend_of_script')
