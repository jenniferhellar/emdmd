"""
A script to create training and test cross-validation splits on the 
extracted EmDMD features of the requested EEG or iEEG dataset. 

Iterates over the patients specified in const.py and splits the data per
patient to create global training and test data splits that are saved to
the filesystem.
-----------------------------------------------------------------------------

file: split_kfolds.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: os, argparse, pickle, numpy, random, sklearn
local dependencies: const.py

-----------------------------------------------------------------------------
usage: split_kfolds.py [-h] --dataset DATASET [--patient PATIENT] --sph SPH --sop SOP

Creates training/test cross-validation splits on the computed EmDMD features of the input dataset.

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name e.g. chb-mit, kaggle-ieeg
  --patient PATIENT  Patient name e.g. chb01, Patient_1. Default: None
  --sph SPH          Seizure prediction horizon (minutes). Default: 0
  --sop SOP          Seizure onset period (minutes). Default: 5
-----------------------------------------------------------------------------
Detailed description
-----------------------------------------------------------------------------
Creates output directory 'tempdata/[dataset]/folds/'

Seeds the random number generator

Iterates over each patient in the dataset (if no patient is specified)
- Opens each preictal features file and concatenates the data
    - Preictal data is taken from (t_onset - sop) to (t_onset - sph)
- Opens each interictal features file and concatenates the data
- Randomly sub-samples the interictal data to balance the classes
- Combines the patient preictal and interictal data (X)
- Creates corresponding class labels (y); preictal - 1, interictal - 0
- Saves these to files by naming convention
     [patient]_X.pkl, [patient]_y.pkl
- Splits the patient X, y by K-Fold cross-validation
- For each fold, concatenates the appropriate X_tr, y_tr, etc. to the
    global (dataset) CV folds in X_train, y_train, etc.

Saves the global data for each CV fold to files
    fold[fold #]_Xtrain.pkl, fold[fold #]_Xtest.pkl, ...
-----------------------------------------------------------------------------
"""
import os
import argparse
import pickle
import numpy as np
import random

from sklearn.model_selection import KFold

# local script
import const


parser = argparse.ArgumentParser(
    description='Creates training/test cross-validation splits on the computed EmDMD features of the input dataset.')

parser.add_argument('--dataset', required=True,
                    help='Dataset name e.g. chb-mit, kaggle-ieeg')

parser.add_argument('--patient', required=False, default=None,
                    help='Patient name e.g. chb01, Patient_1. \
                    Default: None')

parser.add_argument('--sph', required=True, default=0,
                    help='Seizure prediction horizon (minutes). \
                    Default: 0')

parser.add_argument('--sop', required=True, default=5,
                    help='Seizure onset period (minutes). \
                    Default: 5')

parser.add_argument('--verbose', required=False, default=False,
                    help='Print detailed output. \
                    Default: False')

args = parser.parse_args()
dataset = vars(args)['dataset']
patient = vars(args)['patient']
sph_min = int(vars(args)['sph'])
sop_min = int(vars(args)['sop'])
verbose = bool(vars(args)['verbose'])

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

PATIENTS = const.PATIENTS[dataset]
if patient != None and patient not in PATIENTS:
    print('\n\nERROR: unsupported patient argument. Select from allowed options: {} or add new options in const.get_patients()\n'.format(PATIENTS))
    exit(1)

if patient != None:
    # single patient requested
    patient_lst = [patient]
else:
    patient_lst = PATIENTS

# directory of segment files and parent directory of output dir 'folds'
_, RESDIR, _ = const.get_dirs(dataset)

OUTDIR = os.path.join(RESDIR, 'folds')
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

K = const.K
W = const.W
SEED = const.SEED

# seed the random number generator
random.seed(a=SEED, version=2)

kf = KFold(n_splits=K, random_state=SEED, shuffle=True)

# preictal data taken from (onset - sop) to (onset - sph)
# convert minutes to # of EmDMD feature segments
sph = int(sph_min*60/W)
sop = int(sop_min*60/W)

# accumulate data across patients
X_train = []
X_test = []

y_train = []
y_test = []

for patient_idx in range(len(patient_lst)):
    patient = patient_lst[patient_idx]
    print('analyzing patient ' + patient)

    fpath = os.path.join(RESDIR, patient, 'features')

    files = os.listdir(fpath)
    files.sort()
    n = len(files)

    pre_files = [i for i in files if i.find('preictal') != -1]

    # extract the desired preictal sections
    # concatenate this patient's preictal features
    for i in range(len(pre_files)):
        file = pre_files[i]
        f = open(os.path.join(fpath, file), 'rb')
        features = pickle.load(f)
        f.close()

        # select the preictal data to later classify
        onset = features.shape[0]
        # the kaggle dataset cuts out the 5 min prior to onset
        if dataset == 'kaggle-ieeg' or dataset == 'kaggle-dog':
            onset += int(5*60/W)
        if i == 0:
            pat_preictal = features[(onset-sop):(onset-sph), :]
        else:
            pat_preictal = np.concatenate((pat_preictal, features[(onset-sop):(onset-sph), :]), axis=0)

    # extract all interictal sections
    inter_files = [i for i in files if i.find('interictal') != -1]
    for i in range(len(inter_files)):
        file = inter_files[i]
        f = open(os.path.join(fpath, file), 'rb')
        features = pickle.load(f)
        f.close()

        if i == 0:
            pat_interictal = features
        else:
            pat_interictal = np.concatenate((pat_interictal, features), axis=0)

    # sub-sample the interictal data to same amount as preictal
    pre_len = pat_preictal.shape[0]
    inter_len = pat_interictal.shape[0]
    sidx = random.sample(range(inter_len), pre_len)
    pat_interictal = pat_interictal[sidx, :]

    # concatenate preictal and interictal segments
    X = np.concatenate((pat_preictal, pat_interictal), axis=0)
    # label the data segments
    y = np.concatenate((np.ones((pre_len,1)), np.zeros((pre_len,1))), axis=0)
    if verbose:
        print(patient, X.shape, y.shape)

    # write the data to files (for redundancy)
    outfile = '{}_X.pkl'.format(patient)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(X, f)
    f.close()

    outfile = '{}_y.pkl'.format(patient)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(y, f)
    f.close()

    # split the patient data for cross-validation
    fold = 0
    for tr_i, tst_i in kf.split(X):
        X_tr , X_tst = X[tr_i, :], X[tst_i, :]
        y_tr , y_tst = y[tr_i] , y[tst_i]
        if patient_idx == 0:
            # append first data for next CV fold to the list of folds
            X_train.append(X_tr)
            X_test.append(X_tst)
            y_train.append(y_tr)
            y_test.append(y_tst)
        else:
            # concatenate this patient's data to the previous
            X_train[fold] = np.concatenate((X_train[fold], X_tr), axis=0)
            y_train[fold] = np.concatenate((y_train[fold], y_tr), axis=0)
            X_test[fold] = np.concatenate((X_test[fold], X_tst), axis=0)
            y_test[fold] = np.concatenate((y_test[fold], y_tst), axis=0)
        fold += 1
if verbose:
    print('X_train\t', len(X_train), X_train[0].shape)
    print('y_train\t', len(y_train), y_train[0].shape)
    print('X_test\t', len(X_test), X_test[0].shape)
    print('y_test\t', len(y_test), y_test[0].shape)


# save the overall dataset cross-validation splits
if verbose:
    print('\ndumping data to file ...')
for fold in range(K):
    outfile = 'fold{}_Xtrain.pkl'.format(fold)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(X_train[fold], f)
    f.close()
    outfile = 'fold{}_Xtest.pkl'.format(fold)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(X_test[fold], f)
    f.close()
    outfile = 'fold{}_ytrain.pkl'.format(fold)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(y_train[fold], f)
    f.close()
    outfile = 'fold{}_ytest.pkl'.format(fold)
    f = open(os.path.join(OUTDIR, outfile), 'wb')
    pickle.dump(y_test[fold], f)
    f.close()