#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/06/17
# Modified Date: 2022/06/30
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
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import cv2 # OpenCV - make sure the OpenCV version is 4.5

from libs.localfeaturematcher import Resize, LocalFeatureMatcher
from libs.utilities import *

from pdb import set_trace as bp


def GeometricVerificationPerCamera(respath, newrespathdir, agent_id, other_id, opt, query_cam_id, other_cam_id):
    datapath=opt.datapath
    dataset=opt.dataset
    
    lfm = LocalFeatureMatcher(opt)

    filename=os.path.join(respath,'agent{:d}_vpr_res.csv'.format(query_cam_id))
    df = pd.read_csv(filename, sep=';', index_col=False)

    res = []

    for j in tqdm(range(0, df.shape[0])):
        query_id = df.iloc[j,1] # Column 1 is QueryID (0-index based)

        retrieved_id = df.iloc[j,4] # Column 4 is MatchID, the retrieved frame in the second camera (if any)
        if (retrieved_id != -1) and (df.iloc[j,6] > 12):
            matches_fname=os.path.join(respath,'agent{:d}_kps_F{:d}_Q{:d}.txt'.format(other_cam_id, retrieved_id, query_id))

            if os.path.exists(matches_fname):
                outfname = os.path.join(newrespathdir, 'agent{:d}_kps_F{:d}_Q{:d}.txt'.format(query_cam_id,retrieved_id,query_id))
                with open(outfname, 'w') as f:
                    if query_cam_id == 1:
                        f.write('2 {:d}\n'.format(retrieved_id))
                        f.write('{:d} {:d}\n'.format(query_cam_id, query_id))
                    elif query_cam_id == 2:
                        f.write('1 {:d}\n'.format(retrieved_id))
                        f.write('{:d} {:d}\n'.format(query_cam_id, query_id))
                f.close()

                if dataset == 'office':
                    query_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(agent_id), 'rgb', '{:06d}.png'.format(query_id))
                    train_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(other_id), 'rgb', '{:06d}.png'.format(retrieved_id))
                else:
                    query_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(agent_id), 'rgb', '{:06d}.png'.format(query_id+1))
                    train_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(other_id), 'rgb', '{:06d}.png'.format(retrieved_id+1))

                detection_status, ninliers = lfm.RunMAGSACplusplus(query_img_name, train_img_name, matches_fname, outfname)
                    
                q_res = np.array([df.iloc[j,0],df.iloc[j,1],df.iloc[j,2],df.iloc[j,3],df.iloc[j,4], detection_status,df.iloc[j,6], ninliers])
            else:
                print('File {:s} does not exist!'.format(matches_fname))
                q_res = np.array([df.iloc[j,0],df.iloc[j,1],df.iloc[j,2],df.iloc[j,3],df.iloc[j,4], df.iloc[j,5],df.iloc[j,6], df.iloc[j,7]])
        else:
            q_res = np.array([df.iloc[j,0],df.iloc[j,1],df.iloc[j,2],df.iloc[j,3],df.iloc[j,4], df.iloc[j,5],df.iloc[j,6], df.iloc[j,7]])

        res.append(q_res)
    
    header_cols = ['ProcessFrameID', 'QueryID','AgentID','ProcessFrameID2', 'MatchID', 'Status','# matches', '# inliers']
    df1 = pd.DataFrame(np.array(res),columns=header_cols)

    # if opt.SACestimator == 'MAGSAC++':
    #     df1.to_csv(os.path.join(newrespathdir, 'agent{:d}_vpr_res_magsac.csv'.format(query_cam_id)),sep=';',index_label=False, index=False)
    # else:
    #     df1.to_csv(os.path.join(newrespathdir, 'agent{:d}_vpr_res.csv'.format(query_cam_id)),sep=';',index_label=False, index=False)

    df1.to_csv(os.path.join(newrespathdir, 'agent{:d}_vpr_res.csv'.format(query_cam_id)),sep=';',index_label=False, index=False)


def MatchingLocalFeaturesPerCamera(respath, newrespathdir, agent_id, other_id, opt, query_cam_id, other_cam_id):
    datapath=opt.datapath
    dataset=opt.dataset
    
    lfm = LocalFeatureMatcher(opt)

    filename=os.path.join(respath,'agent{:d}_vpr_res.csv'.format(query_cam_id))
    df = pd.read_csv(filename, sep=';', index_col=False)

    res = []

    for j in tqdm(range(0, df.shape[0])):
        query_id = df.iloc[j,1] # Column 1 is QueryID (0-index based)

        retrieved_id = df.iloc[j,4] # Column 4 is MatchID, the retrieved frame in the second camera (if any)
        if retrieved_id != -1:
            # Read the query image as query_img
            # and train image This query image
            # is what you need to find in train image
            if dataset == 'office':
                query_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(agent_id), 'rgb', '{:06d}.png'.format(query_id))
                train_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(other_id), 'rgb', '{:06d}.png'.format(retrieved_id))
            else:
                query_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(agent_id), 'rgb', '{:06d}.png'.format(query_id+1))
                train_img_name = os.path.join(datapath,dataset,'images', 'view{:d}'.format(other_id), 'rgb', '{:06d}.png'.format(retrieved_id+1))

            # outfname = os.path.join(newrespathdir, 'agent{:d}_kps_F{:d}_Q{:d}.txt'.format(query_cam_id,retrieved_id,query_id))
            outfname = os.path.join(newrespathdir, 'agent{:d}_kps_F{:d}_Q{:d}.txt'.format(other_cam_id,retrieved_id,query_id))
            with open(outfname, 'w') as f:
                if query_cam_id == 1:
                    f.write('2 {:d}\n'.format(retrieved_id))
                    f.write('{:d} {:d}\n'.format(query_cam_id, query_id))
                elif query_cam_id == 2:
                    f.write('1 {:d}\n'.format(retrieved_id))
                    f.write('{:d} {:d}\n'.format(query_cam_id, query_id))
            f.close()
            
            if (opt.feature_type == 'RootSIFT') or (opt.feature_type == 'SIFT'):
               detection_status, ninliers =  lfm.RunRootSIFT(query_img_name, train_img_name, outfname)
            elif (opt.feature_type == 'SuperPoint'):
                detection_status, ninliers = lfm.RunSuperPoint(query_img_name, train_img_name, outfname)
            elif (opt.feature_type == 'SuperGlue'):
                detection_status, ninliers = lfm.RunSuperGlue(query_img_name, train_img_name, outfname)
            elif (opt.feature_type == 'dbow') or (opt.feature_type == 'deepbit') or (opt.feature_type == 'netvlad'):
                if opt.SACestimator == 'MAGSAC++':
                    detection_status, ninliers = lfm.RunMAGSACplusplus(query_img_name, train_img_name, outfname)
            else:
                raise Exception('Wrong feature type!')
                
            q_res = np.array([df.iloc[j,0],df.iloc[j,1],df.iloc[j,2],df.iloc[j,3],df.iloc[j,4], detection_status,df.iloc[j,6], ninliers])
        else:
            q_res = np.array([df.iloc[j,0],df.iloc[j,1],df.iloc[j,2],df.iloc[j,3],df.iloc[j,4], df.iloc[j,5],df.iloc[j,6], df.iloc[j,7]])

        res.append(q_res)
    
    header_cols = ['ProcessFrameID', 'QueryID','AgentID','ProcessFrameID2', 'MatchID', 'Status','# matches', '# inliers']
    df1 = pd.DataFrame(np.array(res),columns=header_cols)

    # if opt.SACestimator == 'MAGSAC++':
    #     df1.to_csv(os.path.join(newrespathdir, 'agent{:d}_vpr_res_magsac.csv'.format(query_cam_id)),sep=';',index_label=False, index=False)
    # else:
    #     df1.to_csv(os.path.join(newrespathdir, 'agent{:d}_vpr_res.csv'.format(query_cam_id)),sep=';',index_label=False, index=False)

    df1.to_csv(os.path.join(newrespathdir, 'agent{:d}_vpr_res.csv'.format(query_cam_id)),sep=';',index_label=False, index=False)


def RunPairResults(args, id1, id2):
    datapath=args.datapath
    respath=args.respath
    method=args.method
    ref_global = args.ref_global
    n_runs=args.n_runs
    matching_mode = args.matching_mode
    dataset=args.dataset
    frequency=args.frequency

    print('{:s} {:d}|{:d}'.format(dataset, id1, id2))

    for r in tqdm(range(1,n_runs+1)):
        # respath=os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), ref_global, 'FREQ{:d}'.format(frequency), 'run{:d}'.format(r))
        # newrespathdir = os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, 'FREQ{:d}'.format(frequency), 'run{:d}'.format(r))

        respath=os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), ref_global, 'run{:d}'.format(r))
        newrespathdir = os.path.join(args.respath, '{:s}_{:d}vs{:d}'.format(dataset,id1,id2), method, 'run{:d}'.format(r))

        if not os.path.isdir(newrespathdir):
            os.makedirs(newrespathdir)
            print("Directory '%s' created" %newrespathdir)

        if (method == 'dbow-m') or (method == 'deepbit-m') or (method == 'netvlad-m'):
            if args.SACestimator == 'MAGSAC++':
                GeometricVerificationPerCamera(respath, newrespathdir, id1, id2, args, 1, 2)
                GeometricVerificationPerCamera(respath, newrespathdir, id2, id1, args, 2, 1)
            else:
                raise Exception('Wrong estimator for method DBoW/NetVLAD/DeepBit! Only MAGSAC++ can be used (no RANSAC)!')
        elif (method == 'rootsift') or (method == 'superpoint') or (method == 'superglue'):
            MatchingLocalFeaturesPerCamera(respath, newrespathdir, id1, id2, args, 1, 2)
            MatchingLocalFeaturesPerCamera(respath, newrespathdir, id2, id1, args, 2, 1)
        else:
            raise Exception('Wrong method!')



def Run_M3CAM_2_0_Dataset(opt):
    dataset = opt.dataset

    if (dataset == 'gate') or (dataset == 'courtyard'):
        RunPairResults(opt, 1, 2)
        RunPairResults(opt, 1, 3)
        RunPairResults(opt, 1, 4)
        RunPairResults(opt, 2, 3)
        RunPairResults(opt, 2, 4)
        RunPairResults(opt, 3, 4)
    elif dataset == 'office':
        RunPairResults(opt, 1, 2)
        RunPairResults(opt, 1, 3)
        RunPairResults(opt, 2, 3)
    elif dataset == 'backyard':
        # RunPairResults(opt, 1, 2)
        # RunPairResults(opt, 1, 3)
        RunPairResults(opt, 1, 4)
        RunPairResults(opt, 2, 3)
        # RunPairResults(opt, 3, 4)
        # RunPairResults(opt, 2, 4)


def GetParser():
    parser = argparse.ArgumentParser(
        description='C3OD: Image pair matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='gate', type=str)
    parser.add_argument('--datapath', default='', type=str)
    parser.add_argument('--respath', default='', type=str)

    parser.add_argument('--n_runs', default='5', type=int)
    parser.add_argument('--overlap_th', default='50', type=int)
    # parser.add_argument('--ang_th', default='70', type=int)

    parser.add_argument('--ref_global', default='netvlad', type=str, choices=['dbow','deepbit', 'netvlad'])
    parser.add_argument('--method', default='rootsift', type=str, 
        choices=['dbow','deepbit', 'netvlad','dbow-m','deepbit-m', 'netvlad-m',
        'rootsift','superpoint','superglue','patchnetvlad'])
    parser.add_argument('--matching_mode', default='local', type=str, choices=['local','global'])
    parser.add_argument('--frequency', default='5', type=int)
    
    parser.add_argument('--SACestimator', default='MAGSAC++', type=str, choices=['RANSAC','MAGSAC++'])
    parser.add_argument('--min_num_inliers', default='15', type=int)
    parser.add_argument('--ransacReprojThreshold', default='2.0', type=float)
    parser.add_argument('--confidence', default='0.99', type=float)
    parser.add_argument('--maxIters', default='500', type=int)

    parser.add_argument('--matching_strategy', default='NNDR', type=str, choices=['NNDR'])
    parser.add_argument('--snn_th', default='0.6', type=float)

    parser.add_argument('--max_n_kps', default='1000', type=int)
    parser.add_argument('--feature_type', default='RootSIFT', type=str, choices=['RootSIFT','SIFT','SuperPoint','SuperGlue'])


    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1000,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    return parser



if __name__ == '__main__':
    
    print('Initialising:')
    print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
    print('OpenCV {}'.format(cv2.__version__))

    if not CheckOpenCvVersion():
        exit(1)

    parser = GetParser()
    opt = parser.parse_args()
    opt = Resize(opt)
    
    Run_M3CAM_2_0_Dataset(opt)
