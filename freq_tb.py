import os
import pickle
import numpy as np
import math

import emdmd
import util
import const

# dataset = 'kaggle-ieeg'
# patient = 'Patient_2'

dataset = 'kaggle-dog'
patient = 'Dog_5'

preindices = ['0001']
interindices = ['0001']

# dataset original sampling frequency
ORIG_FS = const.ORIG_FS[dataset]
# channels to include for this patient
CHANNELS = const.CHANNELS[dataset][patient]
# data directory and output directory
DATADIR, RESDIR, _ = const.get_dirs(dataset)

# downsampling factor and effective sampling frequency after downsampling
DS_FACTOR = const.DS_FACTOR[dataset]
FS = const.get_fs(dataset)

W = const.W
R = const.R[dataset]

if dataset.find('kaggle') != -1:
	fpath = os.path.join(DATADIR, patient, patient)
else:
	fpath = os.path.join(DATADIR, patient)
OUTDIR = os.path.join(RESDIR, patient)

print('\n\nProcessing '+ dataset + ' record ' + patient + ' ...\n')

for preidx, interidx in zip(preindices, interindices):
	f = patient + '_preictal_segment_' + preidx + '.mat'

	x_pre , _, _, _, seqidx = util.read_mat(os.path.join(fpath, f))
	print(preidx, seqidx)

	# get the channels to use for this patient
	if f in CHANNELS.keys():
		ch = CHANNELS[f]
	else:
		ch = CHANNELS['general']

	x_pre = util.downsample(x_pre[ch,:], DS_FACTOR)

	cut = math.floor(x_pre.shape[1]/(FS*W))*(FS*W)
	x_pre = x_pre[:, :cut]

	pre_features = emdmd.get_fmax(x_pre, fs=FS, w=W, r=R)

	f = patient + '_interictal_segment_' + interidx + '.mat'
	
	x_inter, _, _, _, seqidx = util.read_mat(os.path.join(fpath, f))
	print(interidx, seqidx)

	# get the channels to use for this patient
	if f in CHANNELS.keys():
		ch = CHANNELS[f]
	else:
		ch = CHANNELS['general']

	x_inter = util.downsample(x_inter[ch,:], DS_FACTOR)

	cut = math.floor(x_inter.shape[1]/(FS*W))*(FS*W)
	x_inter = x_inter[:, :cut]

	inter_features = emdmd.get_fmax(x_inter, fs=FS, w=W, r=R)