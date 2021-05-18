#!/usr/bin/env python
#
# Evaluation script for the training set of the CORSMAL Challenge 
# at Intelligent Sensing Summer School 2020, 1-4 Sep
#
################################################################################## 
#        Author: Alessio Xompero
#         Email: corsmal-challenge@qmul.ac.uk
#
#  Created Date: 2020/08/25
# Modified Date: 2020/08/25
#
# Centre for Intelligent Sensing, Queen Mary University of London, UK
# 
# -----------------------------------------------------------------------------
#
# MIT License
#
# Copyright (c) 2021 CORSMAL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#3
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##################################################################################
#
import os
import csv
import numpy as np
import pandas as pd
from sklearn import metrics
import argparse 
import copy 

from pdb import set_trace as bp

def computeWAFS(gt, est):
	gt = gt.astype(str)
	est = est.astype(str)

	# return (metrics.f1_score(gt, est, average='weighted') * 100).tolist()
	return (metrics.f1_score(gt, est, average='weighted') * 100)

def FillingTypeAndLevelMapping(f_type, f_level):
	f_type_lvl = np.ones(f_type.shape[0]) * -2
	
	idx0 = np.where( (f_type == 0) & (f_level == 0) )
	idx1 = np.where( (f_type == 1) & (f_level == 50) )
	idx2 = np.where( (f_type == 1) & (f_level == 90) )
	idx3 = np.where( (f_type == 2) & (f_level == 50) )
	idx4 = np.where( (f_type == 2) & (f_level == 90) )
	idx5 = np.where( (f_type == 3) & (f_level == 50) )
	idx6 = np.where( (f_type == 4) & (f_level == 90) )
	
	f_type_lvl[idx0] = 0
	f_type_lvl[idx1] = 1
	f_type_lvl[idx2] = 2
	f_type_lvl[idx3] = 3
	f_type_lvl[idx4] = 4
	f_type_lvl[idx5] = 5
	f_type_lvl[idx6] = 6
	
	
	return f_type_lvl

def computeTaskScores(gt, est, test_set, model_name):
	if test_set == 'novel_containers':
		print('Novel containers')
		idx = np.where( (gt['Container ID'].values  > 15) )

	elif test_set == 'ccm_novel':
		print('All testing containers')
		idx = np.where( (gt['Container ID'].values  > 9) )

	elif test_set == 'ccm_public':
		print('Public testing containers (CCM)')
		idx = np.where( (gt['Container ID'].values  > 9) & (gt['Container ID'].values  < 13) )

	elif test_set == 'ccm_private':
		print('Private testing containers (CCM)')
		idx = np.where( (gt['Container ID'].values  > 12) & (gt['Container ID'].values  < 16) )

	elif test_set == 'ccm_testing':
		print('All testing containers (CCM)')
		idx = np.where( (gt['Container ID'].values  > 9) & (gt['Container ID'].values  < 16) )

	elif test_set == 'ccm_val':
		print('Validation containers (CCM)')
		val_ids = pd.read_csv('val_ids.txt')
		val_ids = np.reshape(val_ids.values,(1,val_ids.values.shape[0]), 'C')[0]
		idx1 = np.where( (gt['Container ID'].values <= 9) )
		idx = idx1[0][val_ids]

	elif test_set == 'ccm_train':
		print('Training containers (CCM)')
		train_ids = pd.read_csv('train_ids.txt')
		train_ids = np.reshape(train_ids.values,(1,train_ids.values.shape[0]), 'C')[0]
		idx1 = np.where( (gt['Container ID'].values <= 9) )
		idx = idx1[0][train_ids]
	
	# task = 'Filling type'
	# FT = computeWAFS(gt[task].values[idx], est[task].values[idx])
	
	# task = 'Filling level'
	# FL = computeWAFS(gt[task].values[idx], est[task].values[idx])

	# gt_joint = FillingTypeAndLevelMapping(gt['Filling type'], gt['Filling level'])
	# est_joint = FillingTypeAndLevelMapping(est['Filling type'], est['Filling level'])
	# FTL = computeWAFS(gt_joint[idx], est_joint[idx])
	
	
	AC = computeWAFS(gt['Action'].values[idx], est['action_pred'].values[idx])

	print('Accuracy of the action classifier: {:3.2f}%'.format(AC))

	ComputeAndSaveConfMat(gt['Action'].values[idx], est['action_pred'].values[idx], test_set, model_name)


#-------------------------------------
# Save confusion matrix in LaTex format
def ComputeAndSaveConfMat(gt, est, test_set, model_name):
	conf_mtx = metrics.confusion_matrix(gt, est, normalize='true').astype(np.float)
	# tot_rows = conf_mtx.sum(axis=1).reshape(-1,1)

	# if any(tot_rows == 0):
	# 	nonzeroidx = [i for i, e in enumerate(tot_rows) if e != 0];
	# 	conf_mtx[nonzeroidx] = conf_mtx[nonzeroidx] / tot_rows[nonzeroidx]
	# else:
	# 	conf_mtx = (conf_mtx / conf_mtx.sum(axis=1).reshape(-1,1))
	
	myfile = open('confmat_{:s}_{:s}.txt'.format(test_set,model_name), 'w')
	myfile.write('x y z\n')
	for k in range(0,conf_mtx.shape[0]):
		for l in range(0,conf_mtx.shape[1]):
			myfile.write('{:d} {:d} {:.2f}\n'.format(l, k, conf_mtx[k][l]))
		
		myfile.write('\n')
	myfile.close()


if __name__ == '__main__':
	# Arguments
	parser = argparse.ArgumentParser(description='CORSMAL Challenge evaluation')
	parser.add_argument('--submission', default='submissions/team1.csv', type=str)
	parser.add_argument('--testset', default='ccm_public', type=str, choices=['ccm_train','ccm_val','ccm_private','ccm_public','ccm_testing','novel_containers','ccm_novel'])
	args = parser.parse_args()

	# Read annotations
	gt = pd.read_csv('annotations_w_action.csv', sep=',')
	
	# Read submission
	est = pd.read_csv(args.submission, sep=',')

	model_name = args.submission[12:-4]
	
	est['Filling level'] = est['Filling level'].replace(1,50)
	est['Filling level'] = est['Filling level'].replace(2,90)
	

	computeTaskScores(gt, est, args.testset, model_name)
