import os
import argparse
import pickle
import numpy as np

import emdmd 
from const import RESDIR, FS, R



parser = argparse.ArgumentParser(
    description='Processes a single patient from the MIT-CHB dataset to extract EmDMD features.')

parser.add_argument('-p', '--patient', required=True, default='chb05',
                    help='Patient to process. \
                    Default: chb05')

parser.add_argument('--class', required=False, default=None,
                    help='Segment EEG class to process. \
                    Default: None')

parser.add_argument('--index', required=False, default=None,
                    help='Segment EEG index to process. \
                    Default: None')

args = parser.parse_args()
patient = vars(args)['patient']
fclass = vars(args)['class']
fidx = vars(args)['index']

if fclass == None and fidx == None:
    MULTI_FILE = True
elif fclass == None or fidx == None:
    print('To process a single file, both file class and file index must be specified.')
    exit(0)
else:
    MULTI_FILE = False
    fidx = int(fidx)


fpath = os.path.join(RESDIR, patient)

outDir = os.path.join(RESDIR, patient, 'features')
if not os.path.isdir(RESDIR):
	os.makedirs(RESDIR)
if not os.path.isdir(outDir):
	os.makedirs(outDir)

print('\n\nProcessing MIT-CHB record ' + patient + ' ...\n')

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

print('\n')
print('total preictal windows ... ', n_preictal_windows)
print('total interictal windows ... ', n_interictal_windows)
print('\nFinished!')
