#! /usr/bin/env/ python
#
################################################################################## 
# Author: 
#   Alessio Xompero: a.xompero@qmul.ac.uk
#
#  Created Date: 2022/06/17
# Modified Date: 2022/07/04
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
import pandas as pd
import numpy as np

# from pdb import set_trace as bp


def ConvertResults(opt):
	filename=opt.fnamein
	fnameout = opt.fnameout

	methods = ['dbow','netvlad','deepbit','dbow-m','netvlad-m','deepbit-m',
        'rootsift','superpoint','superglue']

	df = pd.read_csv(filename, sep=',', index_col=False)
	df = df.drop_duplicates()

	with open(fnameout, 'w') as fout:

		seq_pair = df.SequencePair.unique()
		for s in seq_pair:
			print(s)
			
			fout.write('{:s}\n'.format(s))

			res_seq_pair = np.zeros((3,9))
			
			j=0
			for m in methods:
				df1 = df.loc[(df['SequencePair'] == s) & (df['Method'] == m)]

				if len(df1) > 0:
					res_seq_pair[0,j] = df1['P'].values[0]
					res_seq_pair[1,j] = df1['R'].values[0]
					res_seq_pair[2,j] = df1['A'].values[0]

				j += 1

			fout.write('& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n'.format(
				res_seq_pair[0,0],res_seq_pair[0,1],res_seq_pair[0,2],res_seq_pair[0,3],res_seq_pair[0,4], res_seq_pair[0,5],
				res_seq_pair[0,6],res_seq_pair[0,7],res_seq_pair[0,8]))
			fout.write('& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n'.format(
				res_seq_pair[1,0],res_seq_pair[1,1],res_seq_pair[1,2],res_seq_pair[1,3],res_seq_pair[1,4], res_seq_pair[1,5],
				res_seq_pair[1,6],res_seq_pair[1,7],res_seq_pair[1,8]))
			fout.write('& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n'.format(
				res_seq_pair[2,0],res_seq_pair[2,1],res_seq_pair[2,2],res_seq_pair[2,3],res_seq_pair[2,4], res_seq_pair[2,5],
				res_seq_pair[2,6],res_seq_pair[2,7],res_seq_pair[2,8]))

	fout.close()

def GetParser():
    parser = argparse.ArgumentParser(
        description='C3OD: Convert results to LaTex table',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fnamein', default='', type=str)
    parser.add_argument('--fnameout', default='', type=str)
    

    return parser

if __name__ == '__main__':
    
    print('Initialising:')
    print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))

    parser = GetParser()
    opt = parser.parse_args()

    ConvertResults(opt)
    