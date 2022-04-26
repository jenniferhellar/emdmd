"""
A script to label, extract, and save preictal/interictal segments
from the requested EEG or iEEG dataset. Iterates over the patients 
specified in const.py and calls extract_seg.py on each. Creates a
plain text log file per patient.

See extract_seg.py for more details.
-----------------------------------------------------------------------------

file: extract_all_seg.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: os, argparse
local dependencies: const.py

-----------------------------------------------------------------------------
usage: extract_all_seg.py [-h] --dataset DATASET

Extracts preictal/interictal segments from all patients in the input dataset.

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name e.g. chb-mit, kaggle-ieeg
-----------------------------------------------------------------------------
"""
import os
import argparse

# local script
import const


parser = argparse.ArgumentParser(
	description='Extracts preictal/interictal segments from all patients in the input dataset.')

parser.add_argument('--dataset', required=True,
					help='Dataset name e.g. chb-mit, kaggle-ieeg')

parser.add_argument('--sph', required=True, default=0,
					help='Seizure prediction horizon (minutes). \
					Default: 0')

parser.add_argument('--sop', required=True, default=5,
					help='Seizure onset period (minutes). \
					Default: 5')

parser.add_argument('-v', '--verbose', required=False, default=0,
                    help='Verbose output. \
                    Default: 0')

args = parser.parse_args()
dataset = vars(args)['dataset']
sph = int(vars(args)['sph'])
sop = int(vars(args)['sop'])
verbose = int(vars(args)['verbose'])

if dataset not in const.DATASETS:
	print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
	exit(1)

PATIENTS = const.PATIENTS[dataset]
W = const.W

for p in PATIENTS:
	os.system('python3 split_kfolds.py --dataset {} --patient {} --sph {} --sop {}'.format(dataset, p, sph, sop))
	os.system('python3 classify.py --dataset {} --verbose {}'.format(dataset, verbose))
	# os.system('python split_kfolds.py --dataset {} --patient {} --sph {} --sop {}'.format(dataset, p, sph, sop))
	# os.system('python classify.py --dataset {} --verbose {}'.format(dataset, verbose))