#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/06/16
# Modified Date: 2022/06/16
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

    
    def detectAndComputeSIFTfeatures(self, img):
        kps, des = self.extractor.detectAndCompute(img,None)

        return kps, des

    def detectAndComputeRootSIFTfeatures(self, img, eps=1e-7):
        kps, descs = self.ComputeSIFTfeatures(img)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return kps, descs

    def detectAndCompute(self, img):
        if self.method == 'SIFT':
            kps, des = rootsift_extractor.detectAndComputeSIFTfeatures(img)
        elif self.method == 'RootSIFT':
            kps, des = rootsift_extractor.detectAndComputeRootSIFTfeatures(img)

        return kps, des

    def computeRootSIFTfeatures(self, descs, eps=1e-7):
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        # return a tuple of the keypoints and descriptors
        return descs