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
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import linregress

# local script
import const


def get_scores(ytrue, ypred):
    auroc = metrics.roc_auc_score(ytrue, ypred)
    ppv = metrics.precision_score(ytrue, ypred, pos_label=1)
    npv = metrics.precision_score(ytrue, ypred, pos_label=0)
    sen = metrics.recall_score(ytrue, ypred, pos_label=1)
    spec = metrics.recall_score(ytrue, ypred, pos_label=0)
    f1 = metrics.f1_score(ytrue, ypred, pos_label=1)
    kappa = metrics.cohen_kappa_score(ytrue, ypred)

    scores = {'AUROC':auroc, 'PPV':ppv, 'NPV':npv, 'Sensitivity':sen, 'Specificity':spec,
                'F1':f1, 'Kappa':kappa}
    return scores

def get_calibration(Xtest, ytest, clf, classifier_type):
    if classifier_type == 'svm-uncal':
        probs = clf.decision_function(Xtest)
        cal_y, cal_x = calibration_curve(ytest, probs, n_bins=10, normalize=True)
    else:
        probs = clf.predict_proba(Xtest)[:, 1]
        cal_y, cal_x = calibration_curve(ytest, probs, n_bins=10)
    slope, intercept, r_value, p_value, std_err = linregress(cal_x, cal_y)
    calibration = {'CalSlope':slope, 'CalIntercept':intercept, 'CalR2':r_value**2}
    return calibration

def print_all(res):
    print('----------------------------------------------------------------------------------')
    print('{:<30}{}'.format('# Segments', res['Seg']))
    print('{:<30}{}'.format('# Features', res['Feat']))
    print('{:<30}{}'.format('AUROC', res['AUROC']))
    print('{:<30}{}'.format('F1', res['F1']))
    print('{:<30}{}'.format('Kappa', res['Kappa']))
    # print('{:<30}{}'.format('PPV', res['PPV']))
    # print('{:<30}{}'.format('NPV', res['NPV'])
    print('{:<30}{}'.format('Sensitivity', res['Sensitivity']))
    print('{:<30}{}'.format('Specificity', res['Specificity']))
    print('{:<30}{}'.format('Cal Slope', res['CalSlope']))
    print('{:<30}{}'.format('Cal Intercept', res['CalIntercept']))
    print('{:<30}{}'.format('Cal R2', res['CalR2']))

    print('----------------------------------------------------------------------------------')
    print('{:<30}'.format(res['Seg']))
    print('{:<30}'.format(res['Feat']))
    print('{:<30}'.format(res['AUROC']))
    print('{:<30}'.format(res['F1']))
    print('{:<30}'.format(res['Kappa']))
    # print('{:<30}'.format(res['PPV']))
    # print('{:<30}'.format(res['NPV']))
    print('{:<30}'.format(res['Sensitivity']))
    print('{:<30}'.format(res['Specificity']))
    print('{:<30}'.format(res['CalSlope']))
    print('{:<30}'.format(res['CalIntercept']))
    print('{:<30}'.format(res['CalR2']))
    print('----------------------------------------------------------------------------------')


parser = argparse.ArgumentParser(
    description='Classifies the test data of the input dataset to measure performance.')

parser.add_argument('--dataset', required=True,
                    help='Dataset name e.g. chb-mit, kaggle-ieeg')

parser.add_argument('-f', '--fold', required=False, default=None,
                    help='Fold to process. \
                    Default: None')

parser.add_argument('-v', '--verbose', required=False, default=0,
                    help='Verbose output. \
                    Default: 0')

args = parser.parse_args()
dataset = vars(args)['dataset']
verbose = int(vars(args)['verbose'])

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

d = {'AUROC':0, 'PPV':0, 'NPV':0, 'Sensitivity':0, 'Specificity':0, 'F1':0, 'Kappa':0, 
     'CalSlope':0, 'CalIntercept':0, 'CalR2':0,
     'Seg':0, 'Feat':0}
results = {'LinSVM':d, 'LinSVM-Cal':d, 'RF':d, 'RF-Cal':d}

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

    # # select the best features according to logistic regression model
    selector = SelectFromModel(estimator=LogisticRegression(max_iter=200)).fit(Xtrain, ytrain)
    feat = selector.get_support()
    Xtrain = Xtrain[:, feat]
    Xtest = Xtest[:, feat]

    # linear SVM
    linSVM = svm.SVC(kernel='linear')
    clf = linSVM.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    res = get_scores(ytest, ypred)
    res.update(get_calibration(Xtest, ytest, clf, 'svm-uncal'))
    res.update({'Seg':(Xtrain.shape[0] + Xtest.shape[0]), 'Feat':Xtrain.shape[1]})
    results['LinSVM'] = {key: res[key]+results['LinSVM'][key] for key in res.keys()}

    # linear SVM, calibrated
    linSVM_Cal = CalibratedClassifierCV(svm.SVC(kernel='linear'), cv=5)
    clf = linSVM_Cal.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    res = get_scores(ytest, ypred)
    res.update(get_calibration(Xtest, ytest, clf, 'svm-cal'))
    res.update({'Seg':(Xtrain.shape[0] + Xtest.shape[0]), 'Feat':Xtrain.shape[1]})
    results['LinSVM-Cal'] = {key: res[key]+results['LinSVM-Cal'][key] for key in res.keys()}


    # random forest
    rf = RandomForestClassifier(n_estimators=100)
    clf = rf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    res = get_scores(ytest, ypred)
    res.update(get_calibration(Xtest, ytest, clf, 'rf'))
    res.update({'Seg':(Xtrain.shape[0] + Xtest.shape[0]), 'Feat':Xtrain.shape[1]})
    results['RF'] = {key: res[key]+results['RF'][key] for key in res.keys()}


    # random forest, calibrated
    rf_Cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100), cv=5)
    rf_Cal.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    res = get_scores(ytest, ypred)
    res.update(get_calibration(Xtest, ytest, clf, 'rf-cal'))
    res.update({'Seg':(Xtrain.shape[0] + Xtest.shape[0]), 'Feat':Xtrain.shape[1]})
    results['RF-Cal'] = {key: res[key]+results['RF-Cal'][key] for key in res.keys()}
    

if MULTI_FOLD:
    print('\nMean CV Metrics')
    print('***********************************************************************************')
    print('Linear SVM, Uncalibrated')
    res = results['LinSVM']
    res = {key: res[key] / K for key in res.keys()}
    if verbose:
        print_all(res)
    print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\'.format('Seg', 'Feat', 'AUROC', 'F1', 'KAP', 'SEN', 'SPE', 'CAS', 'CAI', 'CAR'))
    print('{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\'.format(res['Seg'], res['Feat'], res['AUROC'], res['F1'], res['Kappa'], res['Sensitivity'], res['Specificity'], res['CalSlope'], res['CalIntercept'], res['CalR2']))
    print('\n***********************************************************************************')

    print('\nLinear SVM, Calibrated')
    res = results['LinSVM-Cal']
    res = {key: res[key] / K for key in res.keys()}
    if verbose:
        print_all(res)
    print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\'.format('Seg', 'Feat', 'AUROC', 'F1', 'KAP', 'SEN', 'SPE', 'CAS', 'CAI', 'CAR'))
    print('{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\'.format(res['Seg'], res['Feat'], res['AUROC'], res['F1'], res['Kappa'], res['Sensitivity'], res['Specificity'], res['CalSlope'], res['CalIntercept'], res['CalR2']))
    print('\n***********************************************************************************')

    print('\nRandom Forest, Uncalibrated')
    res = results['RF']
    res = {key: res[key] / K for key in res.keys()}
    if verbose:
        print_all(res)
    print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\'.format('Seg', 'Feat', 'AUROC', 'F1', 'KAP', 'SEN', 'SPE', 'CAS', 'CAI', 'CAR'))
    print('{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\'.format(res['Seg'], res['Feat'], res['AUROC'], res['F1'], res['Kappa'], res['Sensitivity'], res['Specificity'], res['CalSlope'], res['CalIntercept'], res['CalR2']))
    print('\n***********************************************************************************')

    print('\nRandom Forest, Calibrated')
    res = results['RF-Cal']
    res = {key: res[key] / K for key in res.keys()}
    if verbose:
        print_all(res)
    print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\'.format('Seg', 'Feat', 'AUROC', 'F1', 'KAP', 'SEN', 'SPE', 'CAS', 'CAI', 'CAR'))
    print('{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\'.format(res['Seg'], res['Feat'], res['AUROC'], res['F1'], res['Kappa'], res['Sensitivity'], res['Specificity'], res['CalSlope'], res['CalIntercept'], res['CalR2']))
    print('\n***********************************************************************************')

print('\n\n\n')