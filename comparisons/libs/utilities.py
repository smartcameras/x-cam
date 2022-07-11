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

import cv2 # OpenCV - make sure the OpenCV version is 4.5


def ReadMatchedKeypointsFile(filename):
	list_pt1 = []
	list_pt2 = []
	status = []
	dist = []

	with open(filename, 'r') as fin:
		for _ in range(2):
			next(fin)
		for line in fin:
			ll = line.split()

			pt1 = (ll[0], ll[1])
			pt2 = (ll[2], ll[3])

			list_pt1.append(pt1)
			list_pt2.append(pt2)
			status.append(int(ll[4]))
			dist.append(int(-1))

	fin.close()

	return list_pt1, list_pt2, status, dist



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
