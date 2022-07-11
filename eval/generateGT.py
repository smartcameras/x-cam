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

from pdb import set_trace as bp


def ReadAnnotationFile(datapath, id1, id2, filename):
	filepathname = os.path.join(datapath, 'annotation', '{:d}vs{:d}'.format(id1,id2),filename)

	if not os.path.exists(filepathname):
		print('Cannot read ' + filename + "!")
		return 0
	else:
		return  np.loadtxt(filepathname)


def ComputeGroundTruthAnnotation(datapath, id1, id2, overlap_th, ang_th):
	fmt='text' # !!!!!!! To move out of this function !!!!!!!!!!!!!!!!1

	if fmt=='binary':
		### Binary file is larger than the txt file (??)
		outfn=os.path.join(datapath, 'annotation', '{:d}vs{:d}'.format(id1,id2),'groundtruth_{:d}.bin'.format(overlap_th))
	else:
		outfn=os.path.join(datapath, 'annotation', '{:d}vs{:d}'.format(id1,id2),'groundtruth_{:d}.txt'.format(overlap_th))

	if os.path.exists(outfn):
		print('Ground-truth file already exists!')
	else:
		visualhull = ReadAnnotationFile(datapath, id1, id2, 'visualhull.txt')
		angdist = ReadAnnotationFile(datapath, id1, id2, 'angdist.txt')
		trandist = ReadAnnotationFile(datapath, id1, id2, 'trandist.txt')
		
		GT = (visualhull > overlap_th).astype(int)

		GT[np.isnan(visualhull)] = 0
		GT[angdist > ang_th] = 0
		GT[trandist < 0] = 0

		if fmt=='binary':
			GT.tofile(outfn)
		else:
			np.savetxt(outfn, GT, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)


if __name__ == '__main__':

	# Arguments
	parser = argparse.ArgumentParser(description='CO3D Ground-truth generation')
	parser.add_argument('--datapath', default='', type=str)
	parser.add_argument('--overlap_th', default='50', type=int)
	parser.add_argument('--ang_th', default='70', type=int)
	args = parser.parse_args()

	overlap_th=args.overlap_th
	ang_th=args.ang_th

	# gate
	print('Computing ground-truth annotation for gate ...')

	datapath=os.path.join(args.datapath,'gate')
	ComputeGroundTruthAnnotation(datapath, 1, 2, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 1, 3, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 1, 4, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 2, 3, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 2, 4, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 3, 4, overlap_th, ang_th)

	# office
	print('Computing ground-truth annotation for office ...')

	datapath=os.path.join(args.datapath,'office')
	ComputeGroundTruthAnnotation(datapath, 1, 2, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 1, 3, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 2, 3, overlap_th, ang_th)

	# backyard
	print('Computing ground-truth annotation for backyard ...')

	datapath=os.path.join(args.datapath,'backyard')
	# ComputeGroundTruthAnnotation(datapath, 1, 2, overlap_th, ang_th)
	# ComputeGroundTruthAnnotation(datapath, 1, 3, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 1, 4, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 2, 3, overlap_th, ang_th)
	# ComputeGroundTruthAnnotation(datapath, 2, 4, overlap_th, ang_th)
	# ComputeGroundTruthAnnotation(datapath, 3, 4, overlap_th, ang_th)

	# courtyard
	print('Computing ground-truth annotation for courtyard ...')

	datapath=os.path.join(args.datapath,'courtyard')
	ComputeGroundTruthAnnotation(datapath, 1, 2, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 1, 3, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 1, 4, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 2, 3, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 2, 4, overlap_th, ang_th)
	ComputeGroundTruthAnnotation(datapath, 3, 4, overlap_th, ang_th)
	
	print('Finished!')
	