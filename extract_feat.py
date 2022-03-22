import os
import argparse
import pickle
import numpy as np

import emdmd
import const


parser = argparse.ArgumentParser(
    description='Computes EmDMD features from the preictal/interictal segments of the input patient.')

parser.add_argument('--dataset', required=True,
                    help='Dataset name e.g. chb-mit, kaggle-ieeg.')

parser.add_argument('--patient', required=True,
                    help='Patient name e.g. chb01, Patient_1.')

parser.add_argument('--class', required=False, default=None,
                    help='Segment EEG class (preictal/interictal). \
                    Default: None')

parser.add_argument('--index', required=False, default=None,
                    help='Segment EEG index. \
                    Default: None')

args = parser.parse_args()
dataset = vars(args)['dataset']
patient = vars(args)['patient']
fclass = vars(args)['class']
fidx = vars(args)['index']

if dataset not in const.DATASETS:
    print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
    exit(1)

FS, PATIENTS, _ = const.get_details(dataset)
if patient not in PATIENTS:
    print('\n\nERROR: unsupported patient argument. Select from allowed options: {} or add new options in const.get_patients()\n'.format(PATIENTS))
    exit(1)

if fclass == None and fidx == None:
    MULTI_FILE = True
elif fclass == None or fidx == None:
    print('To process a single file, both file class and file index must be specified.')
    exit(0)
else:
    MULTI_FILE = False
    fidx = int(fidx)

_, RESDIR, _ = const.get_dirs(dataset)
_, R, _, _ = const.get_emdmd_params(dataset)

fpath = os.path.join(RESDIR, patient)

outDir = os.path.join(RESDIR, patient, 'features')
if not os.path.isdir(outDir):
	os.makedirs(outDir)

print('\n\nProcessing {} record {} ...'.format(dataset, patient))

if MULTI_FILE:
    files = os.listdir(fpath)
    files = [i for i in files if i.find('.pkl') != -1]
    files.sort()
else:
    files = ['{}_{}{}.pkl'.format(patient, fclass, fidx)]

n_files = len(files)
inter_files = [i for i in files if i.find('interictal') != -1]
n_inter = len(inter_files)
pre_files = [i for i in files if i.find('preictal') != -1]
n_pre = len(pre_files)

n_preictal_windows = 0
n_interictal_windows = 0
i = 0

for file in files:
    print('Extracting features for ' + file + ' ({} of {})'.format(i+1, n_files))
    
    if file.find('preictal') != -1:
        state = 'preictal'
        segidx = file[14:-4]
    elif file.find('interictal') != -1:
        state = 'interictal'
        segidx = file[16:-4]
    else:
        print('\n\nunable to identify state for file ', file, ' ... skipping\n\n')
        continue

    f = open(os.path.join(fpath, file), 'rb')
    seg = pickle.load(f)
    f.close()

    features = emdmd.get_emdmd_features(seg, fs=FS, r=R)
    n = features.shape[0]
    print('\tnumber of windows evaluated ... ', n)

    if state == 'preictal':
        n_preictal_windows += n
    else:
        n_interictal_windows += n

    print('\tdumping data to file ...')
    outfile = patient + '_{}{}_features.pkl'.format(state, segidx)
    f = open(os.path.join(outDir, outfile), 'wb')
    pickle.dump(features, f)
    f.close()

    i += 1

print('\n\nOutput files can be found in {}'.format(outDir))
print('\n')
print('total preictal windows ... ', n_preictal_windows)
print('total interictal windows ... ', n_interictal_windows)
print('\nFinished!')
