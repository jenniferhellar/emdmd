import os
import argparse
import pickle
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn import svm

from const import RESDIR, SEED

k = 5

parser = argparse.ArgumentParser(
    description='Processes the test data from the MIT-CHB dataset to measure performance.')

parser.add_argument('-f', '--fold', required=False, default=None,
                    help='Fold to process. \
                    Default: None')

args = parser.parse_args()

if vars(args)['fold'] == None:
    MULTI_FOLD = True
    folds = [i for i in range(k)]
else:
    MULTI_FOLD = False
    folds = [int(vars(args)['fold'])]

outSubDir = os.path.join(RESDIR, 'folds')
if not os.path.isdir(RESDIR):
    os.makedirs(RESDIR)
if not os.path.isdir(outSubDir):
    os.makedirs(outSubDir)

acc = 0
sen = 0
spec = 0

print('fold', 'accuracy', 'sensitivity', 'specificity')
for i in range(len(folds)):
    fold = folds[i]
    Xtrain_file = 'fold{}_Xtrain.pkl'.format(fold)
    f = open(os.path.join(outSubDir, Xtrain_file), 'rb')
    Xtrain = pickle.load(f)
    f.close()
    # print(Xtrain.shape)

    ytrain_file = 'fold{}_ytrain.pkl'.format(fold)
    f = open(os.path.join(outSubDir, ytrain_file), 'rb')
    ytrain = pickle.load(f)
    f.close()

    Xtest_file = 'fold{}_Xtest.pkl'.format(fold)
    f = open(os.path.join(outSubDir, Xtest_file), 'rb')
    Xtest = pickle.load(f)
    f.close()
    # print(Xtest.shape)

    ytest_file = 'fold{}_ytest.pkl'.format(fold)
    f = open(os.path.join(outSubDir, ytest_file), 'rb')
    ytest = pickle.load(f)
    f.close()

    ytrain = np.reshape(ytrain, (ytrain.shape[0],))
    ytest = np.reshape(ytest, (ytest.shape[0],))


    kernel = 'rbf'
    # parameters from tb_optimize_params.py
    gamma = 1
    C = 10
    classifier = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xtest)
    confusion_mat = confusion_matrix(ytest, y_pred)
    # print(confusion_mat)

    TN = confusion_mat[0][0]
    FP = confusion_mat[0][1]
    FN = confusion_mat[1][0]
    TP = confusion_mat[1][1]
    Accuracy = ((TN + TP)/(TN + FP + FN + TP))
    Precision = (TP / (TP + FP))
    Sensitivity = (TP / (TP +FN))
    Specificity = (TN / (TN + FP))

    print(fold, Accuracy, Sensitivity, Specificity)

    acc += Accuracy
    sen += Sensitivity
    spec += Specificity

if MULTI_FOLD:
    acc = acc / k
    sen = sen / k
    spec = spec / k

    print('avg', acc, sen, spec)