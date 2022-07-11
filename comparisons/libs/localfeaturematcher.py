#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/06/16
# Modified Date: 2022/06/17
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
# import pandas as pd
# from tqdm import tqdm
import random

import cv2 # OpenCV - make sure the OpenCV version is 4.5

from libs.utilities import *
from libs.superglue_matcher import *

from pdb import set_trace as bp


'''
Partially taken from https://pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
'''
class RootSIFT:
    def __init__(self, max_n_kps=-1, method='RootSIFT'):
        # initialize the SIFT feature extractor

        if method not in ['SIFT','RootSIFT']:
            raise Exception('This is the exception you expect to handle')

        self.method = method
        self.max_n_kps = max_n_kps
        
        # self.extractor = cv2.DescriptorExtractor_create("SIFT")
        if max_n_kps == -1:
            self.extractor = cv2.SIFT_create()
        else:
            self.extractor = cv2.SIFT_create(max_n_kps)

    def computeRootSIFTfeatures(self, descs, eps=1e-7):
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return descs

    
    def detectAndComputeSIFTfeatures(self, img):
        kps, des = self.extractor.detectAndCompute(img,None)

        return kps, des

    def detectAndComputeRootSIFTfeatures(self, img, eps=1e-7):
        kps, descs = self.detectAndComputeSIFTfeatures(img)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        descs = self.computeRootSIFTfeatures(descs, eps)
        
        return kps, descs

    def detectAndCompute(self, img):
        if self.method == 'SIFT':
            kps, des = self.detectAndComputeSIFTfeatures(img)
        elif self.method == 'RootSIFT':
            kps, des = self.detectAndComputeRootSIFTfeatures(img)

        return kps, des

class Frame:
    def __init__(self, opt):
        self.img = []
        self.max_n_kps = opt.max_n_kps

        # parameters for SuperPoint/SuperGlue
        self.device = 'cpu' if opt.force_cpu else 'cuda'
        self.resize = opt.resize
        self.resize_float = opt.resize_float

        self.inp = []
        self.scales = []

        # ['SIFT','RootSIFT','SuperPoint','ORB','PatchNetVLAD']
        self.feature_type = opt.feature_type

        if (self.feature_type == 'RootSIFT') or (self.feature_type == 'SIFT'):
            # create sift object
            self.feature_extractor = RootSIFT(self.max_n_kps, self.feature_type)
        elif (self.feature_type == 'SuperPoint') or (self.feature_type == 'SuperGlue'):
            self.feature_extractor = SuperGlueExtractorMatcher(opt)
        else:
            raise Exception('This is the exception you expect to handle! Wrong feature type!')

    def LoadImage(self, imgfilename):

        if not os.path.exists(imgfilename):
            raise Exception('Missing image filename: ' + imgfilename)

        # if (self.feature_type == 'SIFT') or (self.feature_type == 'RootSIFT'):
        self.img = cv2.cvtColor(cv2.imread(imgfilename), cv2.COLOR_BGR2GRAY)
        
        if (self.feature_type == 'SuperPoint') or (self.feature_type == 'SuperGlue'):
            img_superpoint, self.inp, self.scales = read_image(imgfilename, self.device, self.resize, 0, self.resize_float)

    def SetFeatureType(self, method):
        self.feature_type = method

    def SetMaximumNumbefKeypoints(self, max_n_kps):
        self.max_n_kps = max_n_kps

    def ComputeFeatures(self):
        kps, descs = self.feature_extractor.detectAndCompute(self.img)
        return kps, descs

    def GetImage(self):
        return self.img

    def GetSuperGlueLoadImageOutput(self):
        return self.img, self.inp, self.scales
        


class LocalFeatureMatcher:
    def __init__(self, opt):
        self.frame = Frame(opt)

        self.matches = []

        # ['NNDR',]
        self.type = opt.matching_strategy
        self.SNN_threshold = opt.snn_th

        self.flag = opt.SACestimator

        self.usac_params = cv2.UsacParams()

        self.confidence = opt.confidence
        self.maxIters = opt.maxIters
        self.ransacReprojThreshold = opt.ransacReprojThreshold
        self.min_num_inliers = opt.min_num_inliers

        if self.flag == 'RANSAC':
            self.SetRANSACParameters()
        elif self.flag == 'MAGSAC++':
            self.SetMAGSACplusplusParameters()

    # Nearest neighbour matching with Lowe's ratio test (first and second nearest neighbours).
    # Candidate matches are returned sorted by their ratio
    # This function is used for matching SIFT and RootSIFT features
    def NNDRMatcher(self, des1, des2):
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)

        tentatives = []
        # Apply ratio test
        snn_ratios = []
        for m, n in matches:
            if m.distance < self.SNN_threshold * n.distance:
                tentatives.append(m)
                snn_ratios.append(m.distance / n.distance)

        # Sort the points according to the SNN ratio.
        # This step is required both for PROSAC and P-NAPSAC.
        sorted_indices = np.argsort(snn_ratios)
        
        self.matches = list(np.array(tentatives)[sorted_indices])


    def PrintUsacParameters(self, usac_params):
        print('USAC Parameters')

        usac_params = self.usac_params

        print('randomGeneratorState: {:d}'.format(usac_params.randomGeneratorState))
        print('confidence: {:f}'.format(usac_params.confidence))
        print('maxIterations: {:d}'.format(usac_params.maxIterations))
        print('threshold: {:f}'.format(usac_params.threshold))
        print('isParallel: {:d}'.format(int(usac_params.isParallel)))
        print('loIterations: {:d}'.format(usac_params.loIterations))
        print('loSampleSize: {:d}'.format(usac_params.loSampleSize)) 

        if usac_params.score == cv2.SCORE_METHOD_RANSAC:
            print('score: SCORE_METHOD_RANSAC')
        elif usac_params.score == cv2.SCORE_METHOD_MSAC:
            print('score: SCORE_METHOD_MSAC')
        elif usac_params.score == cv2.SCORE_METHOD_MAGSAC:
            print('score: SCORE_METHOD_MAGSAC')
        elif usac_params.score == cv2.SCORE_METHOD_LMEDS:
            print('score: SCORE_METHOD_LMEDS')

        if usac_params.loMethod == cv2.LOCAL_OPTIM_NULL:
            print('loMethod: LOCAL_OPTIM_NULL')
        elif usac_params.loMethod == cv2.LOCAL_OPTIM_INNER_LO:
            print('loMethod: LOCAL_OPTIM_INNER_LO')
        elif usac_params.loMethod == cv2.LOCAL_OPTIM_INNER_AND_ITER_LO:
            print('loMethod: LOCAL_OPTIM_INNER_AND_ITER_LO')
        elif usac_params.loMethod == cv2.LOCAL_OPTIM_GC:
            print('loMethod: LOCAL_OPTIM_GC')
        elif usac_params.loMethod == cv2.LOCAL_OPTIM_SIGMA:
            print('loMethod: LOCAL_OPTIM_SIGMA')

        if usac_params.neighborsSearch == cv2.NEIGH_FLANN_KNN:
            print('neighborsSearch: NEIGH_FLANN_KNN')
        elif usac_params.neighborsSearch == cv2.NEIGH_GRID:
            print('neighborsSearch: NEIGH_GRID')
        elif usac_params.neighborsSearch == cv2.NEIGH_FLANN_RADIUS:
            print('neighborsSearch: NEIGH_FLANN_RADIUS')

        if usac_params.sampler == cv2.SAMPLING_UNIFORM:
            print('sampler: SAMPLING_UNIFORM')
        elif usac_params.sampler == cv2.SAMPLING_PROGRESSIVE_NAPSAC:
            print('sampler: SAMPLING_PROGRESSIVE_NAPSAC')
        elif usac_params.sampler == cv2.SAMPLING_NAPSAC:
            print('sampler: SAMPLING_NAPSAC')
        elif usac_params.sampler == cv2.SAMPLING_PROSAC:
            print('sampler: SAMPLING_PROSAC')

    # set OpenCV USAC parameters for MAGSAC++
    def SetMAGSACplusplusParameters(self):    
        self.usac_params.randomGeneratorState = random.randint(0,1000000)
        self.usac_params.confidence = self.confidence
        self.usac_params.maxIterations = self.maxIters
        self.usac_params.loMethod = cv2.LOCAL_OPTIM_SIGMA
        self.usac_params.score = cv2.SCORE_METHOD_MAGSAC
        self.usac_params.threshold = self.ransacReprojThreshold
        # self.usac_params.isParallel = False # False is deafult
        self.usac_params.loIterations = 10
        self.usac_params.loSampleSize = 50
        self.usac_params.neighborsSearch = cv2.NEIGH_GRID
        self.usac_params.sampler = cv2.SAMPLING_UNIFORM

    # set OpenCV USAC parameters for MAGSAC++
    def SetRANSACParameters(self):    
        self.usac_params.randomGeneratorState = random.randint(0,1000000)
        self.usac_params.confidence = self.confidence
        self.usac_params.maxIterations = self.maxIters
        self.usac_params.loMethod = cv2.LOCAL_OPTIM_NULL
        self.usac_params.score = cv2.SCORE_METHOD_RANSAC
        self.usac_params.threshold = self.ransacReprojThreshold
        # self.usac_params.isParallel = False # False is deafult
        # self.usac_params.loIterations = 10
        # self.usac_params.loSampleSize = 50
        # self.usac_params.neighborsSearch = cv2.NEIGH_FLANN_RADIUS
        self.usac_params.sampler = cv2.SAMPLING_UNIFORM


    def ComputeFundamentalMatrix(self, list_pt1, list_pt2):
        if (len(list_pt1) < self.min_num_inliers) | (len(list_pt2) < self.min_num_inliers):
            return 7, 0, np.zeros((len(list_pt1),1))
        
        vec_pt1 = np.array(list_pt1)
        vec_pt1 = np.float32(vec_pt1[:, np.newaxis, :]).squeeze()

        vec_pt2 = np.array(list_pt2)
        vec_pt2 = np.float32(vec_pt2[:, np.newaxis, :]).squeeze()
        
        # Compute fundamental matrix
        F, status = cv2.findFundamentalMat(vec_pt1, vec_pt2, self.usac_params)
        
        if F is None or F.shape == (1, 1):
            print('No fundamental matrix found')
            return 7, 0, status
        
        if F.shape[0] > 3:
            # more than one matrix found, just pick the first
            print('More than one matrix found')
            print(F)
            F = F[0:3, 0:3]
                
        ninliers =  np.sum(status)
        # print('Number of inliers: {:d}'.format(ninliers))
        
        if ninliers >= self.min_num_inliers:
            return 8, ninliers, status
        else:
            return 7, ninliers, status

    def GetListsOfMatchedKeypoints(self, kps1, kps2, flag=False):
        # extract points
        pts1 = []
        pts2 = []
        dist = []

        for i,(m) in enumerate(self.matches):
            dist.append(m.distance)
            if flag:
                pts2.append(kps2[m.trainIdx].pt)
                pts1.append(kps1[m.queryIdx].pt)
            else:
                pts2.append(kps2[m.trainIdx])
                pts1.append(kps1[m.queryIdx])

        # print('Number of keypoints matched: ({:d}, {:d})'.format(len(pts1),len(pts2)))

        return pts1, pts2, dist

    
    def SaveMatchedFeaturesInliers(self, outfname, list_pt1, list_pt2, status, dist):
        # print('Features saved to file!')
        # results_file = open(outfname+'.txt', 'w')
        results_file = open(outfname, 'w')

        if len(list_pt1) > 1:
            status = status.squeeze()
        # print('{:d} {:d}'.format(len(list_pt1), len(list_pt2)))
        # print('{:d} {:d}'.format(len(status), len(dist)))
        
        # assert(len(list_pt1) == len(list_pt2))
        # assert(len(list_pt1) == len(status))
        # assert(len(list_pt1) == len(dist))
        
        for j in range(len(list_pt1)):
            results_file.write('{:.6f} {:.6f} {:.6f} {:.6f} {:d} {:.6f}\n'.format(float(list_pt1[j][0]),
                float(list_pt1[j][1]),float(list_pt2[j][0]),float(list_pt2[j][1]),int(status[j]),dist[j]))

        results_file.close()


    def PlotMatchesPairImages(self, outfname, query_img, train_img, list_pt1, list_pt2, status):

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
            queryKeypoints.append(cv2.KeyPoint(vec_pt1[i,0],vec_pt1[i,1],0))
            trainKeypoints.append(cv2.KeyPoint(vec_pt2[i,0],vec_pt2[i,1],0))
            if status[i] == 1:
                good_matches.append(cv2.DMatch(i,i,0))

        # draw the matches to the final image
        # containing both the images the drawMatches()
        # function takes both images and keypoints
        # and outputs the matched query image with
        # its train image
        final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, good_matches,None,matchColor=(0,255,0))
        scale_percent = 50 # percent of original size
        width = int(final_img.shape[1] * scale_percent / 100)
        height = int(final_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        final_img = cv2.resize(final_img, dim, interpolation = cv2.INTER_LANCZOS4)

        # Show the final image
        # cv2.imshow("Matches", final_img)
        # cv2.waitKey(500)

        cv2.imwrite(outfname+'.png', final_img)
         


    def RunRootSIFT(self, query_fname, retrieved_fname, outfname):
        self.frame.LoadImage(query_fname)
        kps1, des1 = self.frame.ComputeFeatures()
        img1 = self.frame.GetImage()

        self.frame.LoadImage(retrieved_fname)
        kps2, des2 = self.frame.ComputeFeatures()
        img2 = self.frame.GetImage()

        self.NNDRMatcher(des1, des2)

        if len(self.matches) < 1:
            return 7, 0

        list_pt1, list_pt2, dist = self.GetListsOfMatchedKeypoints(kps1, kps2, True)
        status, ninliers, binliers = self.ComputeFundamentalMatrix(list_pt1, list_pt2)

        self.SaveMatchedFeaturesInliers(outfname, list_pt1, list_pt2, binliers, dist)
        self.PlotMatchesPairImages(outfname, img1, img2, list_pt1, list_pt2, binliers)

        return status, ninliers


    def RunSuperPoint(self, query_fname, retrieved_fname, outfname):
        self.frame.LoadImage(query_fname)
        img0, inp0, scales0 = self.frame.GetSuperGlueLoadImageOutput()

        self.frame.LoadImage(retrieved_fname)
        img1, inp1, scales1 = self.frame.GetSuperGlueLoadImageOutput()

        if img0 is None or img1 is None:
            print('Problem reading image pair: {} {}'.format(query_fname, retrieved_fname))
            exit(1)

        kps0, kps1, des0, des1 = self.frame.feature_extractor.SuperPointExtraction(inp0, inp1)
        self.NNDRMatcher(des0, des1)

        if len(self.matches) < 1:
            return 7, 0

        list_pt1, list_pt2, dist = self.GetListsOfMatchedKeypoints(kps0, kps1, False)
        status, ninliers, binliers = self.ComputeFundamentalMatrix(list_pt1, list_pt2)

        self.SaveMatchedFeaturesInliers(outfname, list_pt1, list_pt2, binliers, dist)
        self.PlotMatchesPairImages(outfname, img0, img1, list_pt1, list_pt2, binliers)

        return status, ninliers



    def RunSuperGlue(self, query_fname, retrieved_fname, outfname):
        self.frame.LoadImage(query_fname)
        img0, inp0, scales0 = self.frame.GetSuperGlueLoadImageOutput()

        self.frame.LoadImage(retrieved_fname)
        img1, inp1, scales1 = self.frame.GetSuperGlueLoadImageOutput()

        if img0 is None or img1 is None:
            print('Problem reading image pair: {} {}'.format(query_fname, retrieved_fname))
            exit(1)  
       
        list_pt1, list_pt2, mconf = self.frame.feature_extractor.SuperGlueMatching(inp0, inp1)
        status, ninliers, binliers = self.ComputeFundamentalMatrix(list_pt1, list_pt2)
        
        self.SaveMatchedFeaturesInliers(outfname, list_pt1, list_pt2, binliers, mconf)
        self.PlotMatchesPairImages(outfname, img0, img1, list_pt1, list_pt2, binliers)

        return status, ninliers



    def RunMAGSACplusplus(self, query_fname, retrieved_fname, infname, outfname):
        self.frame.LoadImage(query_fname)
        img0 = self.frame.GetImage()

        self.frame.LoadImage(retrieved_fname)
        img1 = self.frame.GetImage()

        if img0 is None or img1 is None:
            print('Problem reading image pair: {} {}'.format(query_fname, retrieved_fname))
            exit(1)

        list_pt1, list_pt2, status, dist = ReadMatchedKeypointsFile(infname)
        status, ninliers, binliers = self.ComputeFundamentalMatrix(list_pt1, list_pt2)
        
        self.SaveMatchedFeaturesInliers(outfname, list_pt1, list_pt2, binliers, dist)
        self.PlotMatchesPairImages(outfname, img0, img1, list_pt1, list_pt2, binliers)

        return status, ninliers

class Options:
    def __init__(self):

        self.SACestimator = 'MAGSAC++'
        self.min_num_inliers = 15
        self.ransacReprojThreshold = 2.0
        self.confidence = 0.99
        self.maxIters = 1000
        self.matching_strategy = 'NNDR'
        self.snn_th = 0.6
        self.max_n_kps = 1000
        self.feature_type = 'SuperGlue'

        # SuperGlue/SuperPoint
        self.resize = [640,480]
        self.resize_float = True
        self.superglue='indoor'
        self.max_keypoints = self.max_n_kps
        self.keypoint_threshold = 0.005
        self.nms_radius = 4
        self.sinkhorn_iterations = 20
        self.match_threshold = 0.2
        self.force_cpu = True


def Resize(opt):
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    return opt


def TEST_RootSIFT():
    print('TEST Local Feature Matcher (RootSIFT)')
    opt = Options()

    opt.feature_type = 'RootSIFT'
    
    query_fname = 'samples/000470.png'
    retrieved_fname = 'samples/000480.png'
    outfname = 'rootsift_test_kps_F480_Q470'

    lfm = LocalFeatureMatcher(opt)

    if (opt.feature_type == 'RootSIFT') or (opt.feature_type == 'SIFT'):
        lfm.RunRootSIFT(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperPoint'):
        lfm.RunSuperPoint(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperGlue'):
        lfm.RunSuperGlue(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'dbow') or (opt.feature_type == 'deepbit') or (opt.feature_type == 'netvlad'):
        if opt.SACestimator == 'MAGSAC++':
            lfm.RunMAGSACplusplus(query_fname, retrieved_fname, outfname)
    else:
        raise Exception('Wrong feature type!')

def TEST_SIFT():
    print('TEST Local Feature Matcher (SIFT)')
    opt = Options()

    opt.feature_type = 'SIFT'
    
    query_fname = 'samples/000470.png'
    retrieved_fname = 'samples/000480.png'
    outfname = 'sift_test_kps_F480_Q470'

    lfm = LocalFeatureMatcher(opt)

    if (opt.feature_type == 'RootSIFT') or (opt.feature_type == 'SIFT'):
        lfm.RunRootSIFT(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperPoint'):
        lfm.RunSuperPoint(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperGlue'):
        lfm.RunSuperGlue(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'dbow') or (opt.feature_type == 'deepbit') or (opt.feature_type == 'netvlad'):
        if opt.SACestimator == 'MAGSAC++':
            lfm.RunMAGSACplusplus(query_fname, retrieved_fname, outfname)
    else:
        raise Exception('Wrong feature type!')


def TEST_SuperPoint():
    print('TEST Local Feature Matcher (SuperPoint)')
    opt = Options()

    opt.feature_type = 'SuperPoint'

    opt = Resize(opt)
    
    query_fname = 'samples/000470.png'
    retrieved_fname = 'samples/000480.png'
    outfname = 'superpoint_test_kps_F480_Q470'

    lfm = LocalFeatureMatcher(opt)

    if (opt.feature_type == 'RootSIFT') or (opt.feature_type == 'SIFT'):
        lfm.RunRootSIFT(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperPoint'):
        lfm.RunSuperPoint(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperGlue'):
        lfm.RunSuperGlue(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'dbow') or (opt.feature_type == 'deepbit') or (opt.feature_type == 'netvlad'):
        if opt.SACestimator == 'MAGSAC++':
            lfm.RunMAGSACplusplus(query_fname, retrieved_fname, outfname)
    else:
        raise Exception('Wrong feature type!')



def TEST_SuperGlue():
    print('TEST Local Feature Matcher (SuperGlue)')
    opt = Options()

    opt.feature_type = 'SuperGlue'

    opt = Resize(opt)
    
    query_fname = 'samples/000470.png'
    retrieved_fname = 'samples/000480.png'
    outfname = 'superglue_test_kps_F480_Q470'

    lfm = LocalFeatureMatcher(opt)

    if (opt.feature_type == 'RootSIFT') or (opt.feature_type == 'SIFT'):
        lfm.RunRootSIFT(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperPoint'):
        lfm.RunSuperPoint(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'SuperGlue'):
        lfm.RunSuperGlue(query_fname, retrieved_fname, outfname)
    elif (opt.feature_type == 'dbow') or (opt.feature_type == 'deepbit') or (opt.feature_type == 'netvlad'):
        if opt.SACestimator == 'MAGSAC++':
            lfm.RunMAGSACplusplus(query_fname, retrieved_fname, outfname)
    else:
        raise Exception('Wrong feature type!')

