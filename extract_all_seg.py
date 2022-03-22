import os
import argparse

import const


parser = argparse.ArgumentParser(
    description='Processes all patients from the input dataset to extract preictal and interictal segments.')

parser.add_argument('--dataset', required=True,
					help='Dataset to process.')

args = parser.parse_args()
dataset = vars(args)['dataset']

if dataset not in const.DATASETS:
	print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
	exit(1)

_, PATIENTS, _ = const.get_details(dataset)
_, _, LOGDIR = const.get_dirs(dataset)

for p in PATIENTS:
	logfile = os.path.join(LOGDIR, 'log_seg_{}.txt'.format(p))
	os.system('python3 extract_seg.py --dataset {} --patient {} 2>&1 | tee {}'.format(dataset, p, logfile))