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

args = parser.parse_args()
dataset = vars(args)['dataset']

if dataset not in const.DATASETS:
	print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
	exit(1)

PATIENTS = const.PATIENTS[dataset]
_, _, LOGDIR = const.get_dirs(dataset)

for p in PATIENTS:
	logfile = os.path.join(LOGDIR, 'log_seg_{}.txt'.format(p))
	os.system('python3 extract_seg.py --dataset {} --patient {} 2>&1 | tee {}'.format(dataset, p, logfile))