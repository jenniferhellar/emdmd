import os
import argparse
import pickle
import numpy as np

import emdmd
import const


parser = argparse.ArgumentParser(
    description='Estimates the max/mean r parameter needed for the input patient.')

parser.add_argument('--dataset', required=True,
                    help='Dataset to process.')

parser.add_argument('-p', '--patient', required=True,
                    help='Patient to process.')

args = parser.parse_args()
dataset = vars(args)['dataset']
patient = vars(args)['patient']

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

FS, PATIENTS, _ = const.get_details(dataset)
if patient not in PATIENTS:
    print('\n\nERROR: unsupported patient argument. Select from allowed options: {} or add new options in const.get_patients()\n'.format(PATIENTS))
    exit(1)

_, RESDIR, _ = const.get_dirs(dataset)

fpath = os.path.join(RESDIR, patient)

print('\n\nProcessing {} record {} ...'.format(dataset, patient))

files = os.listdir(fpath)
files.sort()
inter_files = [i for i in files if i.find('interictal') != -1]
pre_files = [i for i in files if i.find('preictal') != -1]

req_len = 1*60*60*FS 	# 1 hour

# if less than one hour available, go to next segment
fidx = 0
m = 0
while m < req_len:
	f = open(os.path.join(fpath, inter_files[fidx]), 'rb')
	X = pickle.load(f)
	f.close()
	m = X.shape[1]
	fidx += 1

print(inter_files[fidx-1])
r_all = emdmd.estimate_r(X, fs=FS)

fidx = 0
m = 0

# if less than one hour available, go to next segment
while m < req_len:
	f = open(os.path.join(fpath, pre_files[fidx]), 'rb')
	X = pickle.load(f)
	f.close()
	m = X.shape[1]
	fidx += 1

print(pre_files[fidx-1])
r = emdmd.estimate_r(X, fs=FS)

r_all = np.concatenate((r_all, r))
r_all = r_all.flatten()

print(r_all.shape[0])
print()
print(np.max(r_all))
print(np.mean(r_all))
print(np.std(r_all))