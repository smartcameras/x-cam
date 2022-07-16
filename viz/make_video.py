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

import os, sys
import argparse
import numpy as np
import pandas as pd

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


class Frame:
	def __init__(self, datapath, imwidth, imheight, new_width):
		self.scale_f = 1.0
		self.datapath = datapath

		self.imwidth = imwidth
		self.imheight = imheight

		self.new_width = new_width
		self.new_height = imheight

		if new_width < imwidth:
			self.scale_f = new_width/imwidth
			self.new_height = int(imheight * self.scale_f)

		self.img = []
		self.matched_img = []
		
		self.black_img = None
		self.init_img = None
		self.noframes_img = None

		self.wait_banner = None
		self.nomatch_banner = None
		self.novalid_banner = None
		self.overlap_banner = None
		self.black_banner = None

		self.LoadMessageImages()

	
	def LoadMessageImages(self):
		for ss in ['init','wait','nomatch','novalid','overlap','noframes','black']:
			fname = os.path.join('imgs','{:s}_{:d}x{:d}.jpg'.format(ss,self.imwidth,self.imheight))
			if not os.path.exists(fname):
				raise Exception('Missing image: ' + fname)

			if ss in ['init','noframes']:
				img_tmp = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
				img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
			else:
				img_tmp = cv2.imread(fname)
			
			if img_tmp.shape[1] > self.new_width:
				height_banner = int(img_tmp.shape[0] * self.scale_f)
				img_tmp2 = cv2.resize(img_tmp, (self.new_width, height_banner), interpolation = cv2.INTER_LANCZOS4)
			else:
				img_tmp2 = img_tmp

			if ss == 'init':
				self.init_img = img_tmp2

			if ss == 'noframes':
				self.noframes_img = img_tmp2

			if ss == 'wait':
				self.wait_banner = img_tmp2

			if ss == 'nomatch':
				self.nomatch_banner = img_tmp2

			if ss == 'novalid':
				self.novalid_banner = img_tmp2

			if ss == 'overlap':
				self.overlap_banner = img_tmp2

			if ss == 'black':
				self.black_banner = img_tmp2
			


	def LoadImage(self, imgfilename):
		fname = os.path.join(self.datapath,imgfilename)
		if not os.path.exists(fname):
			raise Exception('Missing image filename: ' + imgfilename)

		img_tmp = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
		img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
		if img_tmp.shape[1] > self.new_width:
			self.img = cv2.resize(img_tmp, (self.new_width, self.new_height), interpolation = cv2.INTER_LANCZOS4)
		else:
			self.img = img_tmp

		if self.black_img is None:
			self.black_img = np.zeros(self.img.shape, np.uint8)

		return self.img

	def LoadMatchedView(self, imgfilename):
		fname = os.path.join(self.datapath,imgfilename)
		if not os.path.exists(fname):
			raise Exception('Missing image filename: ' + imgfilename)

		img_tmp = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
		img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
		if img_tmp.shape[1] > self.new_width:
			self.matched_img = cv2.resize(img_tmp, (self.new_width, self.new_height), interpolation = cv2.INTER_LANCZOS4)
		else:
			self.matched_img = img_tmp

		return self.matched_img

	def GetBlackImage(self):
		return self.black_img

	def GetInitImage(self):
		return self.init_img

	def GetNoFramesImage(self):
		return self.noframes_img

	def GetWaitBanner(self, mode):
		if mode == 0:
			return cv2.hconcat([self.black_banner,self.wait_banner])
		else:
			return cv2.hconcat([self.wait_banner,self.black_banner])

	def GetNoMatchBanner(self, mode):
		if mode == 0:
			return cv2.hconcat([self.black_banner,self.nomatch_banner])
		else:
			return cv2.hconcat([self.nomatch_banner,self.black_banner])

	def GetNoValidBanner(self, mode):
		if mode == 0:
			return cv2.hconcat([self.black_banner,self.novalid_banner])
		else:
			return cv2.hconcat([self.novalid_banner,self.black_banner])

	def GetOverlapBanner(self, mode):
		if mode == 0:
			return cv2.hconcat([self.black_banner,self.overlap_banner])
		else:
			return cv2.hconcat([self.overlap_banner,self.black_banner])




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


def PlotMatchesPairImages(query_img, train_img, list_pt1, list_pt2, status, scale_width, scale_height):

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

        return final_img


def MakeVideoPair(opt, id1, id2):
	datapath=opt.datapath
	dataset=opt.dataset # gate
	method=opt.method

	init_wnd = 30

	overlap_th=50
	outfn=os.path.join(datapath, dataset, 'annotation', '{:d}vs{:d}'.format(id1,id2),'groundtruth_{:d}.txt'.format(overlap_th))

	if not os.path.exists(outfn):
		print('Cannot read ' + outfn + "!")
		return 0
	else:
		GT = np.loadtxt(outfn)

	### Get query-matching results
	r=1
	respath=os.path.join(opt.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, 'run{:d}'.format(r))

	filename1=os.path.join(respath,'agent1_vpr_res.csv')
	
	if not os.path.exists(filename1):
		print('Cannot read ' + filename1 + "!")
		return 0
	else:
		df1 = pd.read_csv(filename1, sep=';', index_col=False)
		df1 = df1.astype('int32')

	filename2=os.path.join(respath,'agent2_vpr_res.csv')
	if not os.path.exists(filename2):
		print('Cannot read ' + filename2 + "!")
		return 0
	else:	
		df2 = pd.read_csv(filename2, sep=';', index_col=False)
		df2 = df2.astype('int32')

	## Get videos
	if (dataset == 'gate') or (dataset == 'backyard'):
		imwidth = 1280
		imheight = 720
	elif dataset == 'office':
		imwidth = 640
		imheight = 480
	elif dataset == 'courtyard':
		imwidth = 800
		imheight = 450
	
	new_imwidth = 640

	if method == 'superpoint' or method == 'superglue':
		scale_width = 640 / imwidth
		scale_height = 480 / imheight
	else:
		scale_width = 1.0
		scale_height = 1.0
	
	cam1 = Frame(os.path.join(datapath,dataset,'images', 'view{:d}'.format(id1), 'rgb'), imwidth, imheight, new_imwidth)
	cam2 = Frame(os.path.join(datapath,dataset,'images', 'view{:d}'.format(id2), 'rgb'), imwidth, imheight, new_imwidth)

	n_frames_cam1 = len([_ for _ in os.listdir(os.path.join(cam1.datapath)) if _.endswith(r'.png')])
	n_frames_cam2 = len([_ for _ in os.listdir(os.path.join(cam2.datapath)) if _.endswith(r'.png')])

	if dataset == 'office':
		fr=0 # Frame counter
	else:
		fr=1 # Frame counter

	while fr < max(n_frames_cam1, n_frames_cam2):
		print('Frame #{:d}'.format(fr))
		if fr < n_frames_cam1:
			img1 = cam1.LoadImage('{:06d}.png'.format(fr))
		else:
			img1 = cam1.GetNoFramesImage()

		if fr < n_frames_cam2:
			img2 = cam2.LoadImage('{:06d}.png'.format(fr))
		else:
			img2 = cam2.GetNoFramesImage()

		if fr in df1['QueryID'].values:
			match2_id = df1.loc[df1['QueryID'] == fr,'MatchID'].values[0] # Column 4 is MatchID, the retrieved frame in the second camera (if any)
			if match2_id == -1:
				match2_img = cam2.GetBlackImage()
				tmpimg = cv2.hconcat([img1,match2_img])
				tmpbanner = cam2.GetNoMatchBanner(0)
				topimg = cv2.vconcat([tmpimg,tmpbanner])
			else:
				if dataset == 'office':
					query_id = fr
					retrieved_id = match2_id
				else:
					query_id = fr-1
					retrieved_id = match2_id+1
					
				match2_img = cam2.LoadMatchedView('{:06d}.png'.format(retrieved_id))

				infname=os.path.join(respath,'agent2_kps_F{:d}_Q{:d}.txt'.format(match2_id, fr))
				# infname=os.path.join(respath,'agent1_kps_F{:d}_Q{:d}.txt'.format(match2_id, fr)) # rootsift
				list_pt1, list_pt2, status, dist = ReadMatchedKeypointsFile(infname, cam2.scale_f)

				if len(status) == 0:
					tmpimg = cv2.hconcat([img1,match2_img])
				else:
					tmpimg = PlotMatchesPairImages(img1, match2_img, list_pt1, list_pt2, status, scale_width, scale_height)

				if np.sum(np.array(status)) < 12:
					tmpbanner = cam2.GetNoValidBanner(0)
				else:
					tmpbanner = cam2.GetOverlapBanner(0)
				
				topimg = cv2.vconcat([tmpimg,tmpbanner])
		else:
			if fr < init_wnd:
				match2_img = cam2.GetInitImage()
			else:
				match2_img = cam2.GetBlackImage()
			
			tmpimg = cv2.hconcat([img1,match2_img])
			tmpbanner = cam2.GetWaitBanner(0)
			topimg = cv2.vconcat([tmpimg,tmpbanner])

		if fr in df2['QueryID'].values:
			match1_id = df2.loc[df2['QueryID'] == fr,'MatchID'].values[0] # Column 4 is MatchID, the retrieved frame in the second camera (if any)
			if match1_id == -1:
				match1_img = cam1.GetBlackImage()
				tmpimg = cv2.hconcat([match1_img,img2])
				tmpbanner = cam1.GetNoMatchBanner(1)
				botimg = cv2.vconcat([tmpimg,tmpbanner])
			else:
				if dataset == 'office':
					retrieved_id = match1_id
				else:
					retrieved_id = match1_id+1

				match1_img = cam1.LoadMatchedView('{:06d}.png'.format(retrieved_id))

				infname=os.path.join(respath,'agent1_kps_F{:d}_Q{:d}.txt'.format(match1_id, fr))
				# infname=os.path.join(respath,'agent2_kps_F{:d}_Q{:d}.txt'.format(match1_id, fr)) # rootsift
				list_pt1, list_pt2, status, dist = ReadMatchedKeypointsFile(infname, cam1.scale_f)

				if len(status) == 0:
					tmpimg = cv2.hconcat([match1_img,img2])
				else:
					tmpimg = PlotMatchesPairImages(match1_img, img2, list_pt2, list_pt1, status, scale_width, scale_height)

				if np.sum(np.array(status)) < 12:
					tmpbanner = cam1.GetNoValidBanner(1)
				else:
					tmpbanner = cam1.GetOverlapBanner(1)
				
				botimg = cv2.vconcat([tmpimg,tmpbanner])
		else:
			if fr < init_wnd:
				match1_img = cam1.GetInitImage()
			else:
				match1_img = cam1.GetBlackImage()

			tmpimg = cv2.hconcat([match1_img,img2])
			tmpbanner = cam1.GetWaitBanner(1)
			botimg = cv2.vconcat([tmpimg,tmpbanner])


		if len(botimg.shape) == 2:	
			botimg = np.repeat(botimg[:, :, np.newaxis], 3, axis=2)

		if len(topimg.shape) == 2:
			topimg = np.repeat(topimg[:, :, np.newaxis], 3, axis=2)

		frame_banner = np.zeros((100,topimg.shape[1],3), np.uint8)
		top_right_str1 = 'Cross-camera view-overlap recognition'
		top_right_str2 = 'Frame #{:d}'.format(fr)
		cv2.putText(frame_banner, top_right_str1, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), lineType=cv2.LINE_AA)
		cv2.putText(frame_banner, top_right_str2, (topimg.shape[1]-250,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), lineType=cv2.LINE_AA)

		cam_banner = np.zeros((100,topimg.shape[1],3), np.uint8)
		top_left_str = 'Camera {:d}'.format(id1)
		top_right_str = 'Camera {:d}'.format(id2)
		cv2.putText(cam_banner, top_left_str, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), lineType=cv2.LINE_AA)
		cv2.putText(cam_banner, top_right_str, (int(topimg.shape[1]/2)+10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), lineType=cv2.LINE_AA)

		img_tmp1 = cv2.vconcat([frame_banner,cam_banner])
		img_tmp2 = cv2.vconcat([img_tmp1,topimg])
		img_tmp3 = cv2.vconcat([img_tmp2,cam_banner])
		img_k = cv2.vconcat([img_tmp3,botimg])

		if not os.path.isdir(os.path.join(opt.videopath,'{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method)):
			os.makedirs(os.path.join(opt.videopath,'{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method))
			print("Directory '%s' created" %os.path.join(opt.videopath,'{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method))

		cv2.imwrite(os.path.join(opt.videopath,'{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method,'{:06d}.png'.format(fr)), img_k)

		fr += 1


def RunPairResults(args):
	dataset=args.dataset # gate, office, courtyard, backyard

	if (dataset == 'gate') or (dataset == 'courtyard'):
		MakeVideoPair(args, 1, 2)
		MakeVideoPair(args, 1, 3)
		MakeVideoPair(args, 1, 4)
		MakeVideoPair(args, 2, 3)
		MakeVideoPair(args, 2, 4)
		MakeVideoPair(args, 3, 4)
	elif dataset == 'office':
		MakeVideoPair(args, 1, 2)
		MakeVideoPair(args, 1, 3)
		MakeVideoPair(args, 2, 3)
	elif dataset == 'backyard':
		# ComputeCameraPairResults(args, 1, 2)
		# ComputeCameraPairResults(args, 1, 3)
		MakeVideoPair(args, 1, 4)
		MakeVideoPair(args, 2, 3)
		# ComputeCameraPairResults(args, 3, 4)
		# ComputeCameraPairResults(args, 2, 4)



if __name__ == '__main__':
	print('Initialising:')
	print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
	print('OpenCV {}'.format(cv2.__version__))

	if CheckOpenCvVersion():
		# Arguments
		parser = argparse.ArgumentParser(description='CO3D Evaluation')
		parser.add_argument('--dataset', default='gate', type=str)
		parser.add_argument('--datapath', default='', type=str)
		parser.add_argument('--respath', default='', type=str)
		parser.add_argument('--method', default='dbow', type=str, 
			choices=['dbow','deepbit', 'netvlad','dbow-m','deepbit-m', 'netvlad-m','rootsift','superpoint','superglue'])
		parser.add_argument('--videopath', default='tmp', type=str)

		args = parser.parse_args()
		RunPairResults(args)


