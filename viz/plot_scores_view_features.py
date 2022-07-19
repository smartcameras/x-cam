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
import time
import numpy as np

import matplotlib.pyplot as plt

from pdb import set_trace as bp


def ReadQueryScoreResults(filename):

	if not os.path.exists(filename):
		print('File {:s} does not exist!'.format(filename))
		return 

	if os.stat(filename).st_size == 0:
		print('File {:s} is empty!'.format(filename))
		return list_pt1, list_pt2, status, dist

	# to run GUI event loop
	plt.ion()

	# plot
	fig, ax = plt.subplots()

	k=0
	with open(filename, 'r') as fin:
		for line in fin:
			ll = line.split(";")

			l1 = ll[0].split(":")
			query_id = int(l1[1])

			l2 = ll[1].split(":")
			n_matches = int(l2[1])-1

			scores = np.zeros([query_id,2])
			scores[:,0] = np.array(range(0,query_id))
			scores[:,1] = np.NaN

			for j in range(2,n_matches+2):
				ss = ll[j].split(":")
				scores[int(ss[0]),0] = int(ss[0])
				scores[int(ss[0]),1] = float(ss[1])

			# idx = np.argsort(scores[:,0], axis=0, kind='mergesort')
			# scores[:,0] = scores[idx,0]
			# scores[:,1] = scores[idx,1]

			if k == 0:
				line1, = ax.plot(scores[:,0], scores[:,1], linewidth=2.0)
			else:
				# updating data values
				line1.set_xdata(scores[:,0])
				line1.set_ydata(scores[:,1])


			ax.set(xlim=(0, query_id), ylim=(0, 0.05))


			# drawing updated values
			fig.canvas.draw()

			# This will run the GUI event
			# loop until all UI events
			# currently waiting have been processed
			fig.canvas.flush_events()

			time.sleep(0.5)

			k +=1

			# bp()

	fin.close()

	return


if __name__ == '__main__':

	print('Initialising:')
	print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
	
	# Arguments
	parser = argparse.ArgumentParser(description='CrossCamera View-Overlap Recognition: Plot Matches')
	parser.add_argument('--filename', default='agent_queryresults_test.txt', type=str)

	args = parser.parse_args()
	
	# RunPairPlots(args, 1, 2)
	ReadQueryScoreResults(args.filename)