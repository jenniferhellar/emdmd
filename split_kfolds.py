import os
import argparse
import pickle
import numpy as np
import random

from sklearn.model_selection import KFold

import const


parser = argparse.ArgumentParser(
    description='Creates training/test cross-validation splits on the extracted EmDMD features of the input dataset.')

parser.add_argument('--dataset', required=True,
                    help='Dataset name e.g. chb-mit, kaggle-ieeg.')

parser.add_argument('--sph', required=True, default=0,
                    help='Seizure prediction horizon (minutes). \
                    Default: 0')

parser.add_argument('--sop', required=True, default=5,
                    help='Seizure onset period (minutes). \
                    Default: 5')

args = parser.parse_args()
dataset = vars(args)['dataset']
sph_min = float(vars(args)['sph'])
sop_min = int(vars(args)['sop'])

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

_, RESDIR, _ = const.get_dirs(dataset)

outDir = os.path.join(RESDIR, 'folds')
if not os.path.isdir(outDir):
    os.makedirs(outDir)

FS, PATIENTS, _ = const.get_details(dataset)
W, _, K, SEED = const.get_emdmd_params(dataset)
w_sec = W/FS

random.seed(a=SEED, version=2)

kf = KFold(n_splits=K, random_state=SEED, shuffle=True)

# preictal data taken from (onset - sop) to (onset - sph)
# convert minutes to # of EmDMD feature segments
sph = int(sph_min*60/w_sec)
sop = int(sop_min*60/w_sec)

# accumulate data across patients
X_train = []
X_test = []

y_train = []
y_test = []

for patient_idx in range(len(PATIENTS)):
    patient = PATIENTS[patient_idx]
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

        onset = features.shape[0]
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

    # sub-sample the interictal data to same amt as preictal
    pre_len = pat_preictal.shape[0]
    inter_len = pat_interictal.shape[0]
    sidx = random.sample(range(inter_len), pre_len)
    pat_interictal = pat_interictal[sidx, :]

    # concatenate preictal and interictal segments
    X = np.concatenate((pat_preictal, pat_interictal), axis=0)
    # label the data segments
    y = np.concatenate((np.ones((pre_len,1)), np.zeros((pre_len,1))), axis=0)
    print(patient, X.shape, y.shape)

    # write the data to files (for redundancy)
    outfile = '{}_X.pkl'.format(patient)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(X, f)
    f.close()

    outfile = '{}_y.pkl'.format(patient)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(y, f)
    f.close()

    # split the patient data for cross-validation
    fold = 0
    for tr_i, tst_i in kf.split(X):
        X_tr , X_tst = X[tr_i, :], X[tst_i, :]
        y_tr , y_tst = y[tr_i] , y[tst_i]
        if patient_idx == 0:
            X_train.append(X_tr)
            X_test.append(X_tst)
            y_train.append(y_tr)
            y_test.append(y_tst)
        else:
            X_train[fold] = np.concatenate((X_train[fold], X_tr), axis=0)
            y_train[fold] = np.concatenate((y_train[fold], y_tr), axis=0)
            X_test[fold] = np.concatenate((X_test[fold], X_tst), axis=0)
            y_test[fold] = np.concatenate((y_test[fold], y_tst), axis=0)
        fold += 1

print('X_train\t', len(X_train), X_train[0].shape)
print('y_train\t', len(y_train), y_train[0].shape)
print('X_test\t', len(X_test), X_test[0].shape)
print('y_test\t', len(y_test), y_test[0].shape)
 

# save the overall dataset cross-validation splits
print('\ndumping data to file ...')
for fold in range(k):
    outfile = 'fold{}_Xtrain.pkl'.format(fold)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(X_train[fold], f)
    f.close()
    outfile = 'fold{}_Xtest.pkl'.format(fold)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(X_test[fold], f)
    f.close()
    outfile = 'fold{}_ytrain.pkl'.format(fold)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(y_train[fold], f)
    f.close()
    outfile = 'fold{}_ytest.pkl'.format(fold)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(y_test[fold], f)
    f.close()