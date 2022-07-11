#! /usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import numpy as np
import torch

from libs.SuperGlue.models.matching import Matching
from libs.SuperGlue.models.utils import read_image

from pdb import set_trace as bp

torch.set_grad_enabled(False)


class SuperGlueExtractorMatcher:
    def __init__(self, opt):
        print(opt.force_cpu)
        # self.device = 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        self.SNN_threshold = opt.snn_th

        print(self.device)
        self.model = self.LoadSuperGlueModel(opt)

    def LoadSuperGlueModel(self, opt):
        config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }
        
        return Matching(config).eval().to(self.device)

    
    def SuperGlueMatching(self, query_img, train_img):
        # Perform the matching.
        pred = self.model({'image0': query_img, 'image1': train_img})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        # timer.update('matcher')

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # # Write the matches to disk.
        # out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
        #                'matches': matches, 'match_confidence': conf}
        # np.savez(str(matches_path), **out_matches)

        return mkpts0, mkpts1, mconf

    # pred -> keypoints0, scores0, descriptors0, keypoints1, scores1, descriptors1, matches0, matches1, matching_scores0, matching_scores1
    def SuperPointExtraction(self, query_img, train_img):
        # Perform the matching.
        pred = self.model({'image0': query_img, 'image1': train_img})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        des0, des1 = pred['descriptors0'], pred['descriptors1']

        # print('Number of keypoints extracted: ({:d}, {:d})'.format(len(kpts0),len(kpts1)))
        
        return kpts0, kpts1, des0.transpose(), des1.transpose()
    