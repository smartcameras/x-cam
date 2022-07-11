#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/05/25
# Modified Date: 2022/05/25
#
#####################################################################################
# MIT License
#
# Copyright (c) 2022 Alessio
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
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#####################################################################################

import os
import argparse
import numpy as np
import pandas as pd
from csv import writer

from pdb import set_trace as bp



def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def ComputePerformanceMeasures(df, GT):
	TP=0 # True positives
	FP=0 # False positives
	TN=0 # True negatives
	FN=0 # False negatives

	Q = df.shape[0] # Number of total queries performed by the camera
	# R = df[df1['Status'] == 7].shape[0] + df1[((df1['Status'] == 8) | (df1['Status'] == 1))].shape[0]
	M = df[((df['Status'] == 8) | (df['Status'] == 1))].shape[0]

	# df_b = df[((df['Status'] == 8) | (df['Status'] == 1))]
	for j in range(0, df.shape[0]):
		query_id = df.iloc[j,1] # Column 1 is QueryID (0-index based)

		tmp = GT[query_id][0:df.iloc[j,3]+1] # Column 3 is ProcessFrameID2

		if (df.iloc[j,5] == 8) | (df.iloc[j,5]==1):
			retrieved_id = df.iloc[j,4] # Column 4 is MatchID, the retrieved frame in the second camera (if any)
		else:
			retrieved_id = -1
		
		if retrieved_id == -1:
			if np.sum(tmp) > 0:
				FN += 1
			else:
				TN += 1
		else:
			# bp()
			if np.sum(tmp) > 0:
				if tmp[retrieved_id] == 1:
					TP += 1
				else:
					FP += 1
			else:
				FP += 1

		# print('Query #{:d}/{:d} - TP: {:d}, FP: {:d}, TN: {:d}, FN: {:d}'.format(j+1,df.shape[0], TP, FP, TN, FN))

	# Compute precision, recall, and F1-score
	precision = 0
	recall = 0
	fscore = 0 # F1-score
	acc = 0 # accuracy

	if TP > 0:
		if (TP + FP) > 0:
			precision = TP / (TP + FP) 

		if (TP + FN) > 0:
			recall = TP / (TP + FN) 

		fscore = 2 * TP / (2 * TP + FP + FN) 

	if TP + TN > 0:
		acc = (TP + TN) / (TP + FP + TN + FN)

	# print('Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, Accuracy: {:.4f}'.format(precision,recall,fscore,acc))

	return [Q, M, TP, FP, TN, FN, precision, recall, fscore, acc]


### Global-only image matching
#
def ComputePerformanceMeasuresGlobal(df, GT):
	TP=0 # True positives
	FP=0 # False positives
	TN=0 # True negatives
	FN=0 # False negatives

	Q = df.shape[0] # Number of total queries performed by the camera
	# R = df[df1['Status'] == 7].shape[0] + df1[((df1['Status'] == 8) | (df1['Status'] == 1))].shape[0]
	M = df[(df['MatchID'] != -1)].shape[0]

	# df_b = df[((df['Status'] == 8) | (df['Status'] == 1))]
	for j in range(0, df.shape[0]):
		query_id = df.iloc[j,1] # Column 1 is QueryID (0-index based)

		tmp = GT[query_id][0:df.iloc[j,3]+1] # Column 3 is ProcessFrameID2
		retrieved_id = df.iloc[j,4]
		
		if retrieved_id == -1:
			if np.sum(tmp) > 0:
				FN += 1
			else:
				TN += 1
		else:
			# bp()
			if np.sum(tmp) > 0:
				if tmp[retrieved_id] == 1:
					TP += 1
				else:
					FP += 1
			else:
				FP += 1

		# print('Query #{:d}/{:d} - TP: {:d}, FP: {:d}, TN: {:d}, FN: {:d}'.format(j+1,df.shape[0], TP, FP, TN, FN))

	# Compute precision, recall, and F1-score
	precision = 0
	recall = 0
	fscore = 0 # F1-score
	acc = 0 # accuracy

	if TP > 0:
		if (TP + FP) > 0:
			precision = TP / (TP + FP) 

		if (TP + FN) > 0:
			recall = TP / (TP + FN) 

		fscore = 2 * TP / (2 * TP + FP + FN) 

	if TP + TN > 0:
		acc = (TP + TN) / (TP + FP + TN + FN)

	# print('Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, Accuracy: {:.4f}'.format(precision,recall,fscore,acc))

	return [Q, M, TP, FP, TN, FN, precision, recall, fscore, acc]


def ComputeCombinedPerformanceScores(res1, res2):
		res12 = np.array(res1[2:6]) + np.array(res2[2:6])
		
		TP = res12[0]
		FP = res12[1]
		TN = res12[2]
		FN = res12[3]
		
		P12=0
		R12=0
		F12=0
		A12=0

		if TP > 0:
			if (TP + FP) > 0:
				P12 = TP / (TP + FP)

			if (TP + FN) > 0:
				R12 = TP / (TP + FN)
		
			F12 = 2 * TP / (2 * TP + FP + FN)
		
		if TP + TN > 0:
			A12 = (TP + TN) / (TP + FP + TN + FN)


		return [P12, R12, F12, A12]


def ComputeRunStatistics(accuracies, fname, fname2):
		A = np.array(accuracies)
		U = np.unique(A)
		V = np.array([len(A[A == x]) for x in U])
		Z = np.vstack((U,V))
		Z[1,:] = Z[1,:] / np.sum(Z[1,:])
		Z = Z.transpose()
		
		S = np.array([np.array([m,np.min(A[0:m]),np.max(A[0:m]),np.mean(A[0:m]),np.median(A[0:m])]) for m in [5,10,25,30,50,100]])

		# numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
		np.savetxt(fname, S, fmt='%.2f', delimiter=' & ', header='Run & Min & Max & Mean & Median')
		np.savetxt(fname2, Z, fmt='%.2f', delimiter=' ', header='X Y')


def ComputeCameraPairResults(args, id1, id2):
	datapath=args.datapath
	respath=args.respath
	method=args.method
	overlap_th=args.overlap_th
	n_runs=args.n_runs
	matching_mode = args.matching_mode
	dataset=args.dataset # gate

	outfn=os.path.join(datapath, dataset, 'annotation', '{:d}vs{:d}'.format(id1,id2),'groundtruth_{:d}.txt'.format(overlap_th))

	if not os.path.exists(outfn):
		print('Cannot read ' + outfn + "!")
		return 0
	else:
		GT = np.loadtxt(outfn)

		myres = np.zeros([n_runs,24])

		for r in range(1,n_runs+1):
			respath=os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, 'run{:d}'.format(r))
			
			filename1=os.path.join(respath,'agent1_vpr_res.csv')
			filename2=os.path.join(respath,'agent2_vpr_res.csv')

			df1 = pd.read_csv(filename1, sep=';', index_col=False)		
			df2 = pd.read_csv(filename2, sep=';', index_col=False)

			df1 = df1.astype('int32')
			df2 = df2.astype('int32')

			if matching_mode == 'local':
				res1 = ComputePerformanceMeasures(df1, GT)
				res2 = ComputePerformanceMeasures(df2, GT.transpose())
			elif matching_mode == 'global':
				res1 = ComputePerformanceMeasuresGlobal(df1, GT)
				res2 = ComputePerformanceMeasuresGlobal(df2, GT.transpose())

			outfilename = 'results_{:s}.csv'.format(matching_mode)

			res12 = ComputeCombinedPerformanceScores(res1, res2)
			myres[r-1,:] = np.array(res1+res2+res12)

		# print(myres)

		myres[:,6:10] *= 100
		myres[:,16:24] *= 100

		header_cols=['Q1','M1','TP1','FP1', 'TN1', 'FN1', 'P1', 'R1', 'F1', 'A1','Q2','M2','TP2','FP2', 'TN2', 'FN2', 'P2', 'R2', 'F2', 'A2','P12', 'R12', 'F12', 'A12']
		df1 = pd.DataFrame(myres,columns=header_cols)
		df1.to_csv(os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, outfilename),
			sep=';',float_format='%.2f',index_label=False, index=False)

		# Compute statistics and distribution
		# dstpath=os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, 'FREQ5')
		# fname1='acc_runs_stats_{:s}_{:s}.txt'.format(matching_mode,sac_mode)
		# fname2='distribution_{:s}_{:s}_{:s}_{:s}.txt'.format(dataset,method,matching_mode,sac_mode)
		# ComputeRunStatistics(df1['A12'].values, os.path.join(dstpath,fname1), os.path.join(dstpath,fname2))
		

		row_contents = ['{:s}_{:d}vs{:d}'.format(dataset,id1,id2),method,
			'{:.2f}'.format(np.median(myres[:,-4])),
			'{:.2f}'.format(np.median(myres[:,-3])),
			'{:.2f}'.format(np.median(myres[:,-1]))]

		# Append a list as new line to an old csv file
		append_list_as_row(os.path.join(args.respath,'results.csv'), row_contents)


def RunPairResults(args):
	dataset=args.dataset # gate, office, courtyard, backyard

	if not os.path.exists(os.path.join(args.respath,'results.csv')):
		with open(os.path.join(args.respath,'results.csv'), 'w', newline='') as write_obj:
			# Create a writer object from csv module
			csv_writer = writer(write_obj)

			# Add contents of list as last row in the csv file
			csv_writer.writerow(['SequencePair', 'Method', 'P', 'R', 'A'])
	
	if (dataset == 'gate') or (dataset == 'courtyard'):
		ComputeCameraPairResults(args, 1, 2)
		ComputeCameraPairResults(args, 1, 3)
		ComputeCameraPairResults(args, 1, 4)
		ComputeCameraPairResults(args, 2, 3)
		ComputeCameraPairResults(args, 3, 4)
		ComputeCameraPairResults(args, 2, 4)
	elif dataset == 'office':
		ComputeCameraPairResults(args, 1, 2)
		ComputeCameraPairResults(args, 1, 3)
		ComputeCameraPairResults(args, 2, 3)
	elif dataset == 'backyard':
		# ComputeCameraPairResults(args, 1, 2)
		# ComputeCameraPairResults(args, 1, 3)
		ComputeCameraPairResults(args, 1, 4)
		ComputeCameraPairResults(args, 2, 3)
		# ComputeCameraPairResults(args, 3, 4)
		# ComputeCameraPairResults(args, 2, 4)



def ComputeAnalysisSequencePair(args, id1, id2, mode):
	datapath=args.datapath
	respath=args.respath
	method=args.method
	overlap_th=args.overlap_th
	n_runs=args.n_runs
	matching_mode = args.matching_mode
	sac_mode = args.sac
	dataset=args.dataset # gate

	outfn=os.path.join(datapath, dataset, 'annotation', '{:d}vs{:d}'.format(id1,id2),'groundtruth_{:d}.txt'.format(overlap_th))

	if not os.path.exists(outfn):
		print('Cannot read ' + outfn + "!")
		return 0
	else:
		GT = np.loadtxt(outfn)

		if mode == 'frequency':
			scheduling = [5]
			# scheduling = [5,7,10,12,15,17,20]
			prefix_dir = 'FREQ'
			lowcase_prefix_dir = 'frequency'
		elif mode == 'initwnd':
			scheduling = [5,10,15,20,25,30,35,40,45,50,75]
			prefix_dir = 'INIT_WND'
			lowcase_prefix_dir = 'initwnd'
		elif mode == 'rate':
			scheduling = [1,5,10,15,25,30]
			prefix_dir = 'RATE'
			lowcase_prefix_dir = 'rate'

		analysis=np.zeros([len(scheduling),4])
		queries=np.zeros([len(scheduling),4])
		analysis_f1score=np.zeros([len(scheduling),4])

		j=0
		for f in scheduling:
			myres = np.zeros([n_runs,24])

			for r in range(1,n_runs+1):
				respath=os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, '{:s}{:d}'.format(prefix_dir,f), 'run{:d}'.format(r))
			
				filename1=os.path.join(respath,'agent1_vpr_res.csv')
				df1 = pd.read_csv(filename1, sep=';', index_col=False)

				filename2=os.path.join(respath,'agent2_vpr_res.csv')
				df2 = pd.read_csv(filename2, sep=';', index_col=False)
				
				if matching_mode == 'local':
					res1 = ComputePerformanceMeasures(df1, GT)
					res2 = ComputePerformanceMeasures(df2, GT.transpose())
					outfilename = 'results_{:d}.csv'.format(overlap_th)
				elif matching_mode == 'global':
					res1 = ComputePerformanceMeasuresGlobal(df1, GT)
					res2 = ComputePerformanceMeasuresGlobal(df2, GT.transpose())
					outfilename = 'results_{:d}_global.csv'.format(overlap_th)

				res12 = ComputeCombinedPerformanceScores(res1, res2)
				myres[r-1,:] = np.array(res1+res2+res12)

			myres[:,6:10] *= 100
			myres[:,16:24] *= 100

			header_cols=['Q1','M1','TP1','FP1', 'TN1', 'FN1', 'P1', 'R1', 'F1', 'A1','Q2','M2','TP2','FP2', 'TN2', 'FN2', 'P2', 'R2', 'F2', 'A2','P12', 'R12', 'F12', 'A12']
			df1 = pd.DataFrame(myres,columns=header_cols)
			df1.to_csv(os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, '{:s}{:d}'.format(prefix_dir,f),outfilename),
				sep=';',float_format='%.2f',index_label=False, index=False)

			queries[j,:] = np.array([f,myres[0,0],myres[0,10],myres[0,0]+myres[0,10]])
			analysis[j,:] = np.array([f,np.min(myres[:,-1]),np.max(myres[:,-1]),np.median(myres[:,-1])])
			analysis_f1score[j,:] = np.array([f,np.min(myres[:,-2]),np.max(myres[:,-2]),np.median(myres[:,-2])])

			# runs_res = np.zeros([n_runs,4])
			# for rr in range(1,n_runs+1):
			# 	runs_res[rr-1, :]= [rr,np.min(myres[:rr,-1]),np.max(myres[:rr,-1]),np.median(myres[:rr,-1])]

			# np.set_printoptions(precision = 3)
			# print(runs_res[:,1:])
			# bp()


			j+=1

		### Save analysis of the accuracy
		filename1 = '{:s}_analysis_accuracy_{:d}_{:s}.csv'.format(lowcase_prefix_dir,overlap_th, matching_mode)
		header_cols=['FREQ','Amin','Amax','Amedian']
		df2 = pd.DataFrame(analysis,columns=header_cols)
		df2.to_csv(os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, filename1),
			sep=';',float_format='%.2f',index_label=False, index=False)

		### Save queries
		filename2 = '{:s}_queries_{:d}_{:s}.csv'.format(lowcase_prefix_dir,overlap_th, matching_mode)
		header_cols=['FREQ','Q1','Q2','Qtot']
		df3 = pd.DataFrame(queries,columns=header_cols)
		df3.to_csv(os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, filename2),
			sep=';',float_format='%.2f',index_label=False, index=False)

		### Save analysis of the accuracy
		filename3 = '{:s}_analysis_f1score_{:d}_{:s}.csv'.format(lowcase_prefix_dir,overlap_th, matching_mode)
		header_cols=['FREQ','Fmin','Fmax','Fmedian']
		df2 = pd.DataFrame(analysis_f1score,columns=header_cols)
		df2.to_csv(os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, filename3),
			sep=';',float_format='%.2f',index_label=False, index=False)





def RunAnalysis(args):
	dataset=args.dataset # gate, office, courtyard, backyard
	mode = args.analysis_mode
	print(mode)
	
	if (dataset == 'gate') or (dataset == 'courtyard'):
		ComputeAnalysisSequencePair(args, 1, 2, mode)
		ComputeAnalysisSequencePair(args, 1, 3, mode)
		ComputeAnalysisSequencePair(args, 1, 4, mode)
		ComputeAnalysisSequencePair(args, 2, 3, mode)
		ComputeAnalysisSequencePair(args, 2, 4, mode)
		ComputeAnalysisSequencePair(args, 3, 4, mode)
	elif dataset == 'office':
		ComputeAnalysisSequencePair(args, 1, 2, mode)
		ComputeAnalysisSequencePair(args, 1, 3, mode)
		ComputeAnalysisSequencePair(args, 2, 3, mode)
	elif dataset == 'backyard':
		# ComputeAnalysisSequencePair(args, 1, 2, mode)
		# ComputeAnalysisSequencePair(args, 1, 3, mode)
		ComputeAnalysisSequencePair(args, 1, 4, mode)
		ComputeAnalysisSequencePair(args, 2, 3, mode)
		# ComputeAnalysisSequencePair(args, 3, 4, mode)
		# ComputeAnalysisSequencePair(args, 2, 4, mode)





if __name__ == '__main__':

	# Arguments
	parser = argparse.ArgumentParser(description='CO3D Evaluation')
	parser.add_argument('--dataset', default='gate', type=str)
	parser.add_argument('--datapath', default='', type=str)
	parser.add_argument('--respath', default='', type=str)
	parser.add_argument('--overlap_th', default='50', type=int)
	# parser.add_argument('--ang_th', default='70', type=int)
	parser.add_argument('--method', default='dbow', type=str, 
		choices=['dbow','deepbit', 'netvlad','dbow-m','deepbit-m', 'netvlad-m','rootsift','superpoint','superglue'])
	parser.add_argument('--n_runs', default='5', type=int)
	parser.add_argument('--matching_mode', default='local', type=str, choices=['local','global'])

	parser.add_argument('--analysis_mode', default='frequency', type=str, choices=['frequency','initwnd','rate','none'])
	
	args = parser.parse_args()
	
	if args.analysis_mode == 'none':
		RunPairResults(args)
	else:
		RunAnalysis(args)

