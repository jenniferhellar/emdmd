import pyedflib
import numpy as np

from scipy import signal, io


def read_edf(filename):
    """
    See https://pyedflib.readthedocs.io/en/latest/ for
    latest documentation of the PyEDFlib toolbox.
    """
    fname=filename
    f = pyedflib.EdfReader(fname)

    # number of channels
    n = f.signals_in_file

    # n x m array (channels x timestep)
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs


def read_mat(filename):
    """
    """
    mat = io.loadmat(filename)
    dkey = [k for k in mat.keys() if k.find('segment') != -1]
    dkey = dkey[0]

    data = mat[dkey]
    data = data[0][0]

    # the actual subject iEEG data in ch x time
    X = data[0]
    # print(X.shape)  # 15 x 3000000

    # time (in seconds)
    t_sec = data[1].flatten()[0]
    # print(t_sec)

    # sampling frequency (in Hz)
    fs = data[2].flatten()[0]
    # print(fs)

    # list of channel names present
    channels = data[3].flatten()
    channels = [ch[0] for ch in channels]
    # print(channels)

    # current segment index
    sequence_idx = data[4].flatten()[0]
    # print(sequence_idx)

    return X, t_sec, fs, channels, sequence_idx


def get_file_summary(patientfile, summaryfile):
    """
    summaryfile: path to summary file
    patientfile: name of patient file to extract info for

    returns: start and stop time of seizure in patientfile

    note that it assumes only one seizure per file
    """
    
    foundfile = False
    with open(summaryfile,'r') as fID:
        for line in fID:
            if line.find(patientfile) != -1:
                foundfile = True
                continue
            if foundfile:
                if line.find('Seizure Start Time') != -1:
                    linsplit = line.split(' ')
                    lst = [i.strip() for i in linsplit if i.strip() != '']
                    start = int(lst[-2])
                if line.find('Seizure End Time') != -1:
                    linsplit = line.split(' ')
                    lst = [i.strip() for i in linsplit if i.strip() != '']
                    stop = int(lst[-2])
                    foundfile = False
    return (start, stop)


def read_summary(summaryfile):

    infodict = {}
    
    with open(summaryfile,'r') as fID:
        for line in fID:
            if line.find('File Name:') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                # print(lst)
                edf = lst[-1]
                infodict[edf] = {}
            if line.find('Number of Seizures in File:') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                seizures = int(lst[-1])
                infodict[edf]['s'] = seizures
                cnt = 0
            if line.find('Seizure') != -1 and line.find('Start Time') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                start = int(lst[-2])*256
            if line.find('Seizure') != -1 and line.find('End Time') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                stop = int(lst[-2])*256
                infodict[edf][cnt] = (start, stop)
                cnt += 1
    return infodict


def butter_bandpass_filter(data, fs, lowcut=0.1, highcut=40, order = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = signal.butter(order, [low, high], btype='bandpass', analog=False)
    y = signal.filtfilt(i, u, data, padlen=150)
    return y


def downsample(X, factor):
    if factor > 1:
        X_out = np.zeros((X.shape[0], int(X.shape[1]/factor)))
        for x in range(0, X.shape[1], factor):
            X_out[:, int(x/factor)] = np.mean(X[:, x:x+factor], axis=1)
    else:
        X_out = X
    return X_out


def read_feature_extraction_log(logfile):

    with open(logfile, 'r') as fID:
        for line in fID:
            if line.find('total preictal windows') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                n_preictal_windows = int(lst[-1])
            if line.find('total interictal windows') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                n_interictal_windows = int(lst[-1])
    return n_preictal_windows, n_interictal_windows