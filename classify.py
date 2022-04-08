"""
A script to measure performance by classification on the cross-validation
fold(s) of the requested EEG or iEEG dataset. 

Iterates over the CV folds, performs feature selection, training, and
classification, and prints relevant performance metrics.
-----------------------------------------------------------------------------

file: classify.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: os, argparse, pickle, numpy, sklearn
local dependencies: const.py

-----------------------------------------------------------------------------
usage: classify.py [-h] --dataset DATASET [-f FOLD]

Classifies the test data of the input dataset to measure performance.

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name e.g. chb-mit, kaggle-ieeg
  -f FOLD, --fold FOLD  Fold to process. Default: None
-----------------------------------------------------------------------------
Detailed description
-----------------------------------------------------------------------------
Reads the CV data files in directory 'tempdata/[dataset]/folds/'

Iterates over each CV fold (if no fold index is specified)
- Normalizes/standardizes each feature of the training/test data
    (mean = 0, std = 1)
- Trains a LogisticRegression model and uses only its support to reduce
    the number of features.
- Trains a linear SVM.
- Classifies the test data and computes performance metrics.

Prints performance metrics per CV fold and averaged across folds.
-----------------------------------------------------------------------------
"""
import os
import argparse
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix

# local script
import const

parser = argparse.ArgumentParser(
    description='Classifies the test data of the input dataset to measure performance.')

parser.add_argument('--dataset', required=True,
                    help='Dataset name e.g. chb-mit, kaggle-ieeg')

parser.add_argument('-f', '--fold', required=False, default=None,
                    help='Fold to process. \
                    Default: None')

args = parser.parse_args()
dataset = vars(args)['dataset']

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

K = const.K

if vars(args)['fold'] == None:
    MULTI_FOLD = True
    folds = [i for i in range(K)]
else:
    MULTI_FOLD = False
    folds = [int(vars(args)['fold'])]

_, RESDIR, _ = const.get_dirs(dataset)

fpath = os.path.join(RESDIR, 'folds')

Accuracy = 0
Sensitivity = 0
Specificity = 0

print('\n\n')
print('fold', 'accuracy', 'sensitivity', 'specificity')
print('\n')
for i in range(len(folds)):
    fold = folds[i]
    Xtrain_file = 'fold{}_Xtrain.pkl'.format(fold)
    f = open(os.path.join(fpath, Xtrain_file), 'rb')
    Xtrain = pickle.load(f)
    f.close()
    # print(Xtrain.shape)

    ytrain_file = 'fold{}_ytrain.pkl'.format(fold)
    f = open(os.path.join(fpath, ytrain_file), 'rb')
    ytrain = pickle.load(f)
    f.close()

    Xtest_file = 'fold{}_Xtest.pkl'.format(fold)
    f = open(os.path.join(fpath, Xtest_file), 'rb')
    Xtest = pickle.load(f)
    f.close()
    # print(Xtest.shape)

    ytest_file = 'fold{}_ytest.pkl'.format(fold)
    f = open(os.path.join(fpath, ytest_file), 'rb')
    ytest = pickle.load(f)
    f.close()

    ytrain = np.reshape(ytrain, (ytrain.shape[0],))
    ytest = np.reshape(ytest, (ytest.shape[0],))

    # normalize/standardize each feature
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # select the best features according to logistic regression model
    selector = SelectFromModel(estimator=LogisticRegression()).fit(Xtrain, ytrain)
    feat = selector.get_support()
    Xtrain = Xtrain[:, feat]
    Xtest = Xtest[:, feat]

    # train the linear SVM
    classifier = svm.SVC(kernel='linear')
    clf.fit(Xtrain, ytrain)

    # classify the test data
    ypred = clf.predict(Xtest)

    # compute performance metrics
    confusion_mat = confusion_matrix(ytest, ypred)
    TP = confusion_mat[0][0]
    FP = confusion_mat[0][1]
    FN = confusion_mat[1][0]
    TN = confusion_mat[1][1]

    acc = (TN + TP) / (TN + FP + FN + TP)
    sen = TP / (TP + FN)
    spec = TN / (TN + FP)

    print('fold: ', fold, '\tacc: ', acc, '\tsen: ', sen, '\tspec: ', spec)

    Accuracy += acc
    Sensitivity += sen
    Specificity += spec

if MULTI_FOLD:
    print('\n\nMean CV Metrics')
    print('accuracy:\t', Accuracy/K)
    print('sensitivity:\t', Sensitivity/K)
    print('specificity:\t', Specificity/K)

print('\n\n')