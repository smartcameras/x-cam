#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/06/17
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

from libs.localfeaturematcher import LocalFeatureMatcher, TEST_SIFT, TEST_RootSIFT, TEST_SuperPoint, TEST_SuperGlue
from libs.utilities import *

from pdb import set_trace as bp

import torch
# device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'


if __name__ == '__main__':
    
    print('Initialising:')
    print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
    print('OpenCV {}'.format(cv2.__version__))

    if not CheckOpenCvVersion():
        exit(1)

    # parser = argparse.ArgumentParser(
    #     description='C3OD: Image pair matching with RootSIFT',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument('--SACestimator', default='MAGSAC++', type=str, choices=['RANSAC','MAGSAC++'])
    # parser.add_argument('--min_num_inliers', default='15', type=int)
    # parser.add_argument('--ransacReprojThreshold', default='2.0', type=float)
    # parser.add_argument('--confidence', default='0.99', type=float)
    # parser.add_argument('--maxIters', default='500', type=int)

    # parser.add_argument('--matching_strategy', default='NNDR', type=str, choices=['NNDR'])
    # parser.add_argument('--snn_th', default='0.6', type=float)

    # parser.add_argument('--max_n_kps', default='1000', type=int)
    # parser.add_argument('--feature_type', default='SuperPoint', type=str, choices=['RootSIFT','SIFT','SuperPoint','SuperGlue'])

    # parser.add_argument(
    #     '--resize', type=int, nargs='+', default=[640, 480],
    #     help='Resize the input image before running inference. If two numbers, '
    #          'resize to the exact dimensions, if one number, resize the max '
    #          'dimension, if -1, do not resize')
    # parser.add_argument(
    #     '--resize_float', action='store_true',
    #     help='Resize the image after casting uint8 to float')

    # parser.add_argument(
    #     '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    #     help='SuperGlue weights')
    # parser.add_argument(
    #     '--max_keypoints', type=int, default=1000,
    #     help='Maximum number of keypoints detected by Superpoint'
    #          ' (\'-1\' keeps all keypoints)')
    # parser.add_argument(
    #     '--keypoint_threshold', type=float, default=0.005,
    #     help='SuperPoint keypoint detector confidence threshold')
    # parser.add_argument(
    #     '--nms_radius', type=int, default=4,
    #     help='SuperPoint Non Maximum Suppression (NMS) radius'
    #     ' (Must be positive)')
    # parser.add_argument(
    #     '--sinkhorn_iterations', type=int, default=20,
    #     help='Number of Sinkhorn iterations performed by SuperGlue')
    # parser.add_argument(
    #     '--match_threshold', type=float, default=0.2,
    #     help='SuperGlue match threshold')
    # parser.add_argument(
    #     '--force_cpu', action='store_true', default=True,
    #     help='Force pytorch to run in CPU mode.')

    # opt = parser.parse_args()

    # if len(opt.resize) == 2 and opt.resize[1] == -1:
    #     opt.resize = opt.resize[0:1]
    # if len(opt.resize) == 2:
    #     print('Will resize to {}x{} (WxH)'.format(
    #         opt.resize[0], opt.resize[1]))
    # elif len(opt.resize) == 1 and opt.resize[0] > 0:
    #     print('Will resize max dimension to {}'.format(opt.resize[0]))
    # elif len(opt.resize) == 1:
    #     print('Will not resize images')
    # else:
    #     raise ValueError('Cannot specify more than two integers for --resize')


    # query_fname = 'samples/000470.png'
    # retrieved_fname = 'samples/000480.png'
    # outfname = 'test_kps_F480_Q470.txt'

    # lfm = LocalFeatureMatcher(opt)
    
    # if (opt.feature_type == 'RootSIFT') or (opt.feature_type == 'SIFT'):
    #     lfm.RunRootSIFT(query_fname, retrieved_fname, outfname)
    # elif (opt.feature_type == 'SuperPoint'):
    #     lfm.RunSuperPoint(query_fname, retrieved_fname, outfname)
    # elif (opt.feature_type == 'SuperGlue'):
    #     lfm.RunSuperGlue(query_fname, retrieved_fname, outfname)
    # elif (opt.feature_type == 'dbow') or (opt.feature_type == 'deepbit') or (opt.feature_type == 'netvlad'):
    #     if opt.SACestimator == 'MAGSAC++':
    #         lfm.RunMAGSACplusplus(query_fname, retrieved_fname, outfname)
    # else:
    #     raise Exception('Wrong feature type!')

    TEST_SIFT()
    TEST_RootSIFT()
    TEST_SuperPoint()
    TEST_SuperGlue()
