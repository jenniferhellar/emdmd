import os
import argparse
import pickle
import numpy as np
import random

from sklearn.model_selection import GridSearchCV
from sklearn import svm
 
# from const import RESDIR, SEED
import const



def param_selection(X, y, classifier, nfolds, verbosity=0):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=nfolds, verbose=verbosity)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_, grid_search.best_score_


parser = argparse.ArgumentParser(
    description='Processes training data for the input dataset to optimize SVM parameters.')

parser.add_argument('--dataset', required=True,
                    help='Dataset to process.')

parser.add_argument('-f', '--fold', required=False, default=None,
                    help='Fold to process. If None, processes all folds. \
                    Default: None')

args = parser.parse_args()
dataset = vars(args)['dataset']
fold = vars(args)['fold']

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

if fold != None:
    fold = int(vars(args)['fold'])

_, RESDIR, _ = const.get_dirs(dataset)

outDir = os.path.join(RESDIR, 'folds')
if not os.path.isdir(outDir):
    os.makedirs(outDir)

_, _, K, SEED = const.get_emdmd_params(dataset)

random.seed(a=SEED, version=2)

print('accuracy', 'gamma', 'C', 'segments')

if fold != None:

    X_file = 'fold{}_Xtrain.pkl'.format(fold)
    y_file = 'fold{}_ytrain.pkl'.format(fold)

    f = open(os.path.join(outDir, X_file), 'rb')
    X = pickle.load(f)
    f.close()

    f = open(os.path.join(outDir, y_file), 'rb')
    y = pickle.load(f)
    f.close()

    y = np.reshape(y, (y.shape[0],))

    kernel = 'rbf'
    classifier = svm.SVC(kernel=kernel)
    best_params, best_score = param_selection(X, y, classifier, nfolds = K, verbosity=0)

    gamma = best_params['gamma']
    C = best_params['C']
    print(best_score, gamma, C, y.shape[0])

else:

    for fold in range(k):
        X_file = 'fold{}_Xtrain.pkl'.format(fold)
        y_file = 'fold{}_ytrain.pkl'.format(fold)

        f = open(os.path.join(outDir, X_file), 'rb')
        X = pickle.load(f)
        f.close()

        f = open(os.path.join(outDir, y_file), 'rb')
        y = pickle.load(f)
        f.close()

        y = np.reshape(y, (y.shape[0],))

        kernel = 'rbf'
        classifier = svm.SVC(kernel=kernel)
        best_params, best_score = param_selection(X, y, classifier, nfolds = K, verbosity=0)

        gamma = best_params['gamma']
        C = best_params['C']
        print(best_score, gamma, C, y.shape[0])