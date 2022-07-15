#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/06/01
# Modified Date: 2022/06/01
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
import sys
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

import cv2 # OpenCV - make sure the OpenCV version is 4.5

from pdb import set_trace as bp

def CheckOpenCvVersion():
	(major, minor, _) = cv2.__version__.split(".")
	
	if (int(major) < 4):
		print('OpenCV version should be at least than 4.5 to run USAC estimators!')
		return False
	elif (int(major) >= 4) & (int(minor) < 5):
		print('OpenCV version should be at least than 4.5 to run USAC estimators!')
		return False
	else:
		return True

def ReadMatchedKeypointsFile(filename, scale_f):
	list_pt1 = []
	list_pt2 = []
	status = []
	dist = []

	if not os.path.exists(filename):
		print('File {:s} does not exist!'.format(filename))
		return list_pt1, list_pt2, status, dist

	if os.stat(filename).st_size == 0:
		print('File {:s} is empty!'.format(filename))
		return list_pt1, list_pt2, status, dist

	with open(filename, 'r') as fin:
		for _ in range(2):
			next(fin)
		for line in fin:
			ll = line.split()

			pt1 = (float(ll[0])*scale_f, float(ll[1])*scale_f)
			pt2 = (float(ll[2])*scale_f, float(ll[3])*scale_f)

			list_pt1.append(pt1)
			list_pt2.append(pt2)
			status.append(int(ll[4]))
			dist.append(int(-1))

	fin.close()

	return list_pt1, list_pt2, status, dist

def PlotMatches(datapath, dataset, respath, agent_id, other_id, query_cam_id, other_cam_id, method):
	filename=os.path.join(respath,'agent{:d}_vpr_res.csv'.format(query_cam_id))
	
	if not os.path.exists(filename):
		print('File {:s} does not exist!'.format(filename))

	df = pd.read_csv(filename, sep=';', index_col=False)

	if (dataset == 'gate') or (dataset == 'backyard'):
		imwidth = 1280
		imheight = 720
	elif dataset == 'office':
		imwidth = 640
		imheight = 480
	elif dataset == 'courtyard':
		imwidth = 800
		imheight = 450
	else:
		imwidth = 1280
		imheight = 720
	

	for j in range(0, df.shape[0]):
		query_id = int(df.iloc[j,1]) # Column 1 is QueryID (0-index based)
		retrieved_id = int(df.iloc[j,4]) # Column 4 is MatchID, the retrieved frame in the second camera (if any)

		train_agent_id = int(df.iloc[j,3]) # Column 4 is MatchID, the retrieved frame in the second camera (if any)

		if retrieved_id != -1:
			filename=os.path.join(respath,'agent{:d}_kps_F{:d}_Q{:d}.txt'.format(int(other_cam_id), int(retrieved_id), int(query_id)))

			# Read the query image as query_img
			# and train image This query image
			# is what you need to find in train image
			if dataset == 'office':
				query_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(agent_id), 'rgb', '{:06d}.png'.format(query_id))
				train_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(other_id), 'rgb', '{:06d}.png'.format(retrieved_id))
			else:
				query_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(agent_id), 'rgb', '{:06d}.png'.format(query_id+1))
				train_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(other_id), 'rgb', '{:06d}.png'.format(retrieved_id+1))
			
			img_tmp = cv2.imread(query_img_name, cv2.IMREAD_GRAYSCALE)
			query_img = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
			
			img_tmp = cv2.imread(train_img_name, cv2.IMREAD_GRAYSCALE)
			train_img = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)

			if method == 'superpoint' or method == 'superglue':
				scale_width = 640 / imwidth
				scale_height = 480 / imheight
			else:
				scale_width = 1.0
				scale_height = 1.0
			
			list_pt1, list_pt2, status, dist = ReadMatchedKeypointsFile(filename, 1.0)

			if len(status) == 0:
					final_img = cv2.hconcat([query_img,train_img])
			else:

				assert(len(list_pt1) == len(list_pt2))

				vec_pt1 = np.array(list_pt1)
				vec_pt2 = np.array(list_pt2)

				if len(list_pt1) == 1:
					vec_pt1 = np.float32(vec_pt1[:, np.newaxis, :])[0]
					vec_pt2 = np.float32(vec_pt2[:, np.newaxis, :])[0]
				else:
					vec_pt1 = np.float32(vec_pt1[:, np.newaxis, :]).squeeze()
					vec_pt2 = np.float32(vec_pt2[:, np.newaxis, :]).squeeze()
				
				queryKeypoints = []
				trainKeypoints = []
				good_matches = []
				
				for i in range(len(vec_pt1)):
					queryKeypoints.append(cv2.KeyPoint(vec_pt1[i,0]/scale_width,vec_pt1[i,1]/scale_height,0))
					trainKeypoints.append(cv2.KeyPoint(vec_pt2[i,0]/scale_width,vec_pt2[i,1]/scale_height,0))
					if status[i] == 1:
						good_matches.append(cv2.DMatch(i,i,0))

				final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, good_matches,None,matchColor=(0,255,0),singlePointColor=(0,0,255))
				scale_percent = 50 # percent of original size
				width = int(final_img.shape[1] * scale_percent / 100)
				height = int(final_img.shape[0] * scale_percent / 100)
				dim = (width, height)
				final_img = cv2.resize(final_img, dim, interpolation = cv2.INTER_LANCZOS4)

			cv2.imwrite(filename.replace('.txt','.png'), final_img)




def RunPairPlots(args, id1, id2):
	datapath=args.datapath
	respath=args.respath
	method=args.method
	n_runs=args.n_runs
	matching_mode = args.matching_mode
	dataset=args.dataset

	print('{:s} {:d}|{:d}'.format(dataset, id1, id2))

	for r in tqdm(range(1,n_runs+1)):
		respath=os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, 'run{:d}'.format(r))

		PlotMatches(datapath, dataset, respath, id1, id2, 1, 2, method)
		PlotMatches(datapath, dataset, respath, id2, id1, 2, 1, method)	

def Run_M3CAM_2_0_Dataset(opt):
	dataset = opt.dataset

	if (dataset == 'gate') or (dataset == 'courtyard'):
		RunPairPlots(opt, 1, 2)
		RunPairPlots(opt, 1, 3)
		RunPairPlots(opt, 1, 4)
		RunPairPlots(opt, 2, 3)
		RunPairPlots(opt, 2, 4)
		RunPairPlots(opt, 3, 4)
	elif dataset == 'office':
		RunPairPlots(opt, 1, 2)
		RunPairPlots(opt, 1, 3)
		RunPairPlots(opt, 2, 3)
	elif dataset == 'backyard':
		# RunPairPlots(opt, 1, 2)
		# RunPairPlots(opt, 1, 3)
		RunPairPlots(opt, 1, 4)
		RunPairPlots(opt, 2, 3)
		# RunPairPlots(opt, 3, 4)
		# RunPairPlots(opt, 2, 4)


if __name__ == '__main__':

	print('Initialising:')
	print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
	print('OpenCV {}'.format(cv2.__version__))
	
	if CheckOpenCvVersion():
		# Arguments
		parser = argparse.ArgumentParser(description='CrossCamera View-Overlap Recognition: Plot Matches')
		parser.add_argument('--dataset', default='gate', type=str)
		parser.add_argument('--datapath', default='', type=str)
		parser.add_argument('--respath', default='', type=str)
		parser.add_argument('--overlap_th', default='50', type=int)
		# parser.add_argument('--ang_th', default='70', type=int)
		parser.add_argument('--method', default='dbow', type=str, 
			choices=['dbow','deepbit', 'netvlad','dbow-m','deepbit-m', 'netvlad-m','rootsift','superpoint','superglue'])
		parser.add_argument('--n_runs', default='5', type=int)
		parser.add_argument('--matching_mode', default='local', type=str)

		args = parser.parse_args()
		
		# RunPairPlots(args, 1, 2)
		Run_M3CAM_2_0_Dataset(args)