import os
import argparse

import const


parser = argparse.ArgumentParser(
    description='Estimates the r parameter needed for each patient in the input dataset.')

parser.add_argument('--dataset', required=True,
					help='Dataset name e.g. chb-mit, kaggle-ieeg.')

args = parser.parse_args()
dataset = vars(args)['dataset']

if dataset not in const.DATASETS:
	print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
	exit(1)

PATIENTS = const.PATIENTS[dataset]
_, _, LOGDIR = const.get_dirs(dataset)

for p in PATIENTS:
	logfile = os.path.join(LOGDIR, 'log_r_{}.txt'.format(p))
	os.system('python3 estimate_r.py --dataset {} --patient {} 2>&1 | tee {}'.format(dataset, p, logfile))