"""
A script to extract EmDMD features from the preictal/interictal segment(s)
of the requested EEG or iEEG patient.
-----------------------------------------------------------------------------

file: extract_feat.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: os, argparse, pickle, numpy
local dependencies: util.py, const.py, emdmd.py

-----------------------------------------------------------------------------
usage: extract_feat.py [-h] --dataset DATASET --patient PATIENT [--class CLASS] [--index INDEX]

Computes EmDMD features from the preictal/interictal segment(s) of the input patient.

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name e.g. chb-mit, kaggle-ieeg
  --patient PATIENT  Patient name e.g. chb01, Patient_1
  --class CLASS      Segment EEG class (preictal/interictal). Default: None
  --index INDEX      Segment EEG index. Default: None

-----------------------------------------------------------------------------
Detailed description
-----------------------------------------------------------------------------
Extraction and processing:
- Loads required parameters (W, R) from const.py

Creates output directory 'tempdata/[dataset]/[patient]/features/'
- Saves files by naming convention [patient]_[state][index]_features.pkl 
    e.g. chb01_preictal0_features.pkl
-----------------------------------------------------------------------------
"""
import os
import argparse
import pickle
import numpy as np

# local scripts
import emdmd
import const
import util


parser = argparse.ArgumentParser(
    description='Computes EmDMD features from the preictal/interictal segment(s) of the input patient.')

parser.add_argument('--dataset', required=True,
                    help='Dataset name e.g. chb-mit, kaggle-ieeg')

parser.add_argument('--patient', required=True,
                    help='Patient name e.g. chb01, Patient_1')

# optionally specify a single file to process with a class/index pair
parser.add_argument('--class', required=False, default=None,
                    help='Segment EEG class (preictal/interictal). \
                    Default: None')

parser.add_argument('--index', required=False, default=None,
                    help='Segment EEG index. \
                    Default: None')

args = parser.parse_args()
dataset = vars(args)['dataset']
patient = vars(args)['patient']
fclass = vars(args)['class']
fidx = vars(args)['index']

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

PATIENTS = const.PATIENTS[dataset]
if patient not in PATIENTS:
    print('\n\nERROR: unsupported patient argument. Select from allowed options: {} or add new options in const.get_patients()\n'.format(PATIENTS))
    exit(1)

if fclass == None and fidx == None:
    MULTI_FILE = True
elif fclass == None or fidx == None:
    print('To process a single file, both file class (preictal/interictal) and file index (0, 1, ...) must be specified.')
    exit(1)
else:
    MULTI_FILE = False
    fidx = int(fidx)

# effective sampling frequency after any downsampling during segment extraction
FS = const.get_fs(dataset)
# window size (in seconds) per preictal/interictal class label
W = const.W
# SVD truncation parameter
R = const.R[dataset]

# directory of segment files and parent directory of output dir 'features'
_, RESDIR, _ = const.get_dirs(dataset)
fpath = os.path.join(RESDIR, patient)

OUTDIR = os.path.join(RESDIR, patient, 'features')
if not os.path.isdir(OUTDIR):
	os.makedirs(OUTDIR)

MAX_INTER_SEG = int(40*60*60/W)     # 40 hours

print('\n\nProcessing {} record {} ...'.format(dataset, patient))

# if no specific file provided, process all segments for the patient
if MULTI_FILE:
    files = os.listdir(fpath)
    files = [i for i in files if i.find('.pkl') != -1 and i.find('info') == -1]
    # sort by the segment integer index
    idx = [int(i[i.index('l')+1:-4]) for i in files]
    sort_idx = np.argsort(idx)
    files = list(np.array(files)[sort_idx])
else:
    files = ['{}_{}{}.pkl'.format(patient, fclass, fidx)]

n_files = len(files)

# interictal files
inter_files = [i for i in files if i.find('interictal') != -1]
n_inter = len(inter_files)

# preictal files
pre_files = [i for i in files if i.find('preictal') != -1]
n_pre = len(pre_files)

n_preictal_windows = 0
n_interictal_windows = 0
i = 0

enough_inter = False

for file in files:
    if file.find('preictal') != -1:
        state = 'preictal'
        segidx = file[file.find('l')+1:-4]
    elif file.find('interictal') != -1:
        state = 'interictal'
        segidx = file[file.find('l')+1:-4]
    else:
        print('\n\nunable to identify state for file ', file, ' ... skipping\n\n')
        continue

    if state == 'interictal' and enough_inter:
        continue

    print('Extracting features for {} ({} of {})'.format(file, i+1, n_files))
    
    # load an EEG/iEEG segment
    f = open(os.path.join(fpath, file), 'rb')
    seg = pickle.load(f)
    f.close()

    # compute the EmDMD features for the whole segment
    features = emdmd.get_emdmd_features(seg, fs=FS, w=W, r=R)
    n = features.shape[0]
    print('\tnumber of windows evaluated ... ', n)

    if state == 'preictal':
        n_preictal_windows += n
    else:
        n_interictal_windows += n

    # save the result to file system
    print('\tdumping data to file ...')
    outfile = '{}_{}{}_features.pkl'.format(patient, state, segidx)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(features, f)
    f.close()

    i += 1

    if n_interictal_windows > MAX_INTER_SEG:
        enough_inter = True

print('\n\nOutput files can be found in {}'.format(OUTDIR))
print('\n')
print('total preictal windows ... ', n_preictal_windows)
print('total interictal windows ... ', n_interictal_windows)
print('\nFinished!')
