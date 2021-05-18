#!/usr/bin/env python
#
# Evaluation script for the training set of the CORSMAL Challenge 
# at Intelligent Sensing Summer School 2020, 1-4 Sep
#
################################################################################## 
# Author: Alessio Xompero: a.xompero@qmul.ac.uk
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

# P, R, F1, _ = metrics.precision_recall_fscore_support(true,pred)
# Pw, Rw, F1w, _ = metrics.precision_recall_fscore_support(true,pred, average='weighted')
# conf_mtx = metrics.confusion_matrix(true, pred, normalize='true')

# x = np.vstack([P,R,F1])
# y = x.T
# z = np.hstack([Pw,Rw,F1w])

# res = np.vstack([y,z])

def PerClassAccuracy(gt, est):
	P, R, F1, _ = metrics.precision_recall_fscore_support(gt,est, labels=np.unique(gt).tolist())
	return (F1*100).tolist()
	

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
	if os.path.isfile('{:s}_results.txt'.format(test_set)):
		fID = open('{:s}_results.txt'.format(test_set), 'a')
	else:
		fID = open('{:s}_results.txt'.format(test_set), 'w')
		fID.write('Model   | Filling type | Filling level | Filling type and level\n')

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
		val_ids = pd.read_csv('annotations/val_ids.txt')
		val_ids = np.reshape(val_ids.values,(1,val_ids.values.shape[0]), 'C')[0]
		idx1 = np.where( (gt['Container ID'].values <= 9) )
		idx = idx1[0][val_ids]

	elif test_set == 'ccm_train':
		train_ids = pd.read_csv('annotations/train_ids.txt')
		train_ids = np.reshape(train_ids.values,(1,train_ids.values.shape[0]), 'C')[0]
		idx1 = np.where( (gt['Container ID'].values <= 9) )
		idx = idx1[0][train_ids]
	
	task = 'Filling type'
	FT = computeWAFS(gt[task].values[idx], est[task].values[idx])
	
	task = 'Filling level'
	FL = computeWAFS(gt[task].values[idx], est[task].values[idx])

	gt_joint = FillingTypeAndLevelMapping(gt['Filling type'], gt['Filling level'])
	est_joint = FillingTypeAndLevelMapping(est['Filling type'], est['Filling level'])
	FTL = computeWAFS(gt_joint[idx], est_joint[idx])
	
	print('Filling type | Filling level | Filling type and level')
	print('{:2f} | {:2f} | {:2f}'.format(FT, FL, FTL))

	fID.write('{:15s} | {:3.2f} | {:3.2f} | {:3.2f}\n'.format(model_name, FT, FL, FTL))
	fID.close()


if __name__ == '__main__':
	# Arguments
	parser = argparse.ArgumentParser(description='CORSMAL Challenge evaluation')
	parser.add_argument('--submission', default='example/acc.csv', type=str)
	parser.add_argument('--testset', default='ccm_val', type=str, choices=['ccm_train','ccm_val','ccm_private','ccm_public','ccm_testing','novel_containers','ccm_novel'])
	args = parser.parse_args()

	# Read annotations
	gt = pd.read_csv('annotations/annotations.csv', sep=',')
	
	# Read submission
	# est = pd.read_csv('sccnet.csv', sep=',')
	# est = pd.read_csv('random.csv', sep=',')
	est = pd.read_csv(args.submission, sep=',')

	model_name = args.submission[12:-4]
	
	est['Filling level'] = est['Filling level'].replace(1,50)
	est['Filling level'] = est['Filling level'].replace(2,90)
	

	task = 'Filling type'
	
	myfile = open('Accuracy_plot.txt', "w")
	# myfile = open('Accuracy.txt', "w")

	# # Per class accuracy
	# # boolArr = gt['Container ID'].values < 10
	# idx = np.where(gt['Container ID'].values < 10)
	# accFT = PerClassAccuracy(gt['Filling type'].values[idx], est['Filling type'].values[idx])
	# resFT = resFT + accFT 
	# print("PerClassAccuracy - Filling Type")
	# print(resFT)

	

	# myfile.write('Empty, {:.2f} & {:.2f}\n'.format(accFT[0], accFL[0]))
	# myfile.write('Half-full, -- & {:.2f}\n'.format(accFL[1]))
	# myfile.write('Full, -- & {:.2f}\n'.format(accFL[2]))
	# myfile.write('Pasta, {:.2f} & --\n'.format(accFT[1]))
	# myfile.write('Rice, {:.2f} & --\n'.format(accFT[2]))
	# myfile.write('Water, {:.2f} & --\n'.format(accFT[3]))
	# myfile.write('Opaque, {:.2f} & {:.2f}\n'.format(accFT[4], accFL[3]))

	

	# boxes = [7,8,9,12,15]

	# # Accuracy per container
	# for objid in range(1,16):
	# 	print(objid)
		
	# 	res = []

	# 	idx = np.where( (gt['Container ID'].values  == objid) & (gt['Scenario'].values  == 's0'))
	# 	FL = PerClassAccuracy(gt[task].values[idx], est[task].values[idx])
	# 	res = res + FL
	# 	if objid in boxes:
	# 		res.append(-1)

	# 	idx = np.where( (gt['Container ID'].values  == objid) & (gt['Scenario'].values  == 's1'))
	# 	FL = PerClassAccuracy(gt[task].values[idx], est[task].values[idx])
	# 	res = res + FL
	# 	if objid in boxes:
	# 		res.append(-1)

	# 	idx = np.where( (gt['Container ID'].values  == objid) & (gt['Scenario'].values  == 's2'))
	# 	FL = PerClassAccuracy(gt[task].values[idx], est[task].values[idx])
	# 	res = res + FL
	# 	if objid in boxes:
	# 		res.append(-1)

	# 	idx = np.where( (gt['Container ID'].values  == objid) )
	# 	FL = computeWAFS(gt[task].values[idx], est[task].values[idx])
	# 	res.append(FL)

	# 	if task == 'Filling level':
	# 		myfile.write('{:d}, & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n'.format(objid,res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9]))
	# 	else:
	# 		myfile.write('{:d}, & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n'.format(objid,res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9],res[10],res[11],res[12]))

	myfile.write('OBJID \t S0 \t S1 \t S2 \t TOT\n')

	# Accuracy per container
	print('Accuracy per container (CCM)')
	for objid in range(1,16):		
		res = []

		idx = np.where( (gt['Container ID'].values  == objid) & (gt['Scenario'].values  == 's0'))
		FL = computeWAFS(gt[task].values[idx], est[task].values[idx])
		res.append(FL)
		
		idx = np.where( (gt['Container ID'].values  == objid) & (gt['Scenario'].values  == 's1'))
		FL = computeWAFS(gt[task].values[idx], est[task].values[idx])
		res.append(FL)
		
		idx = np.where( (gt['Container ID'].values  == objid) & (gt['Scenario'].values  == 's2'))
		FL = computeWAFS(gt[task].values[idx], est[task].values[idx])
		res.append(FL)

		idx = np.where( (gt['Container ID'].values  == objid) )
		FL = computeWAFS(gt[task].values[idx], est[task].values[idx])
		res.append(FL)

		myfile.write('{:d} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \n'.format(objid,res[0],res[1],res[2],res[3]))
		
	# # Accuracy per scenario
	# idx = np.where((gt['Scenario'].values  == 's0'))
	# FT = computeWAFS(gt['Filling type'].values[idx], est['Filling type'].values[idx])
	# resFT.append(FT)

	# FL = computeWAFS(gt['Filling level'].values[idx], est['Filling level'].values[idx])
	# resFL.append(FL)

	# myfile.write('s0, {:.2f} & {:.2f}\n'.format(FT, FL))

	# idx = np.where((gt['Scenario'].values  == 's1'))
	# FT = computeWAFS(gt['Filling type'].values[idx], est['Filling type'].values[idx])
	# resFT.append(FT)

	# FL = computeWAFS(gt['Filling level'].values[idx], est['Filling level'].values[idx])
	# resFL.append(FL)

	# myfile.write('s1, {:.2f} & {:.2f}\n'.format(FT, FL))


	# # Overall accruacy
	# # boolArr = gt['Container ID'].values < 10
	# # idx = np.where(gt['Container ID'].values  )
	# FT = computeWAFS(gt['Filling type'].values, est['Filling type'].values)
	# resFT.append(FT)

	# FL = computeWAFS(gt['Filling level'].values, est['Filling level'].values)
	# resFL.append(FL)

	# myfile.write('Overall, {:.2f} & {:.2f}\n'.format(FT, FL))
	
	myfile.close()


	computeTaskScores(gt, est, args.testset, model_name)

	
	
	print('Other stuff')
	GTnew = []
	ESnew = []

	idx = np.where( (gt['Container ID'].values  > 9) )
	
	gtft = gt['Filling type'].values[idx]
	gtfl = gt['Filling level'].values[idx]

	esft = est['Filling type'].values[idx]
	esfl = est['Filling level'].values[idx]

	for j in range(0,len(idx[0])):
		FT = gtft[j]
		FL = gtfl[j]

		if (FT == 0) & (FL == 0):
			L = 0
		elif (FT == 1) & (FL == 50):
			L = 1
		elif (FT == 1) & (FL == 90):
			L = 2
		elif (FT == 2) & (FL == 50):
			L = 3
		elif (FT == 2) & (FL == 90):
			L = 4
		elif (FT == 3) & (FL == 50):
			L = 5
		elif (FT == 3) & (FL == 90):
			L = 6

		GTnew.append(L)

		FT = esft[j]
		FL = esfl[j]

		if (FT == 0) & (FL == 0):
			L = 0
		elif (FT == 1) & (FL == 50):
			L = 1
		elif (FT == 1) & (FL == 90):
			L = 2
		elif (FT == 2) & (FL == 50):
			L = 3
		elif (FT == 2) & (FL == 90):
			L = 4
		elif (FT == 3) & (FL == 50):
			L = 5
		elif (FT == 3) & (FL == 90):
			L = 6

		ESnew.append(L)

	conf_mtx = metrics.confusion_matrix(GTnew, ESnew).astype(np.float).T
	tot_rows = conf_mtx.sum(axis=1).reshape(-1,1)

	if any(tot_rows == 0):
		nonzeroidx = [i for i, e in enumerate(tot_rows) if e != 0];
		conf_mtx[nonzeroidx] = conf_mtx[nonzeroidx] / tot_rows[nonzeroidx]
	else:
		conf_mtx = (conf_mtx / conf_mtx.sum(axis=1).reshape(-1,1))
	
	myfile = open('confmat.txt', "w")
	myfile.write('x y z\n')
	for k in range(0,7):
		for l in range(0,7):
			myfile.write('{:d} {:d} {:.2f}\n'.format(l, k, conf_mtx[k][l]))
		
		myfile.write('\n')
	myfile.close()
	# np.savetxt('confmat.txt', conf_mtx, fmt='%.4f', delimiter='\t', newline='\n', header='', footer='', comments='# ', encoding=None)