"""
Common utility library.
-----------------------------------------------------------------------------

file: util.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: pyedflib, numpy, scipy

-----------------------------------------------------------------------------
"""
import pyedflib
import numpy as np

from scipy import signal, io


def read_edf(filename):
    """
    Reads an EDF file. See https://pyedflib.readthedocs.io/en/latest/ for
    latest documentation of the PyEDFlib toolbox.

    Arguments:
        filename: full path to EDF file with file name

    Returns:
        n channels x m timesteps numpy data array
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
    Reads a MAT file from the Kaggle-iEEG dataset.

    Arguments:
        filename: full path to MAT file with file name

    Returns:
        X: n channels x m timesteps numpy data array
        t_sec: length in seconds of data
        fs: sampling frequency
        channels: list of channel names
        sequence_idx: current segment index
    """
    # loads as a dictionary
    mat = io.loadmat(filename)
    # find the one key that has all the actual data in it
    dkey = [k for k in mat.keys() if k.find('segment') != -1]
    dkey = dkey[0]

    # get the data and flatten the extra dimensions
    #   (now a tuple of variables)
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
    For a given CHB-MIT patient data file, retrive the seizure info
    from the patient summary file.

    Arguments:
        patientfile: name of patient file
        summaryfile: full path to and name of summary file

    Returns: 
        Tuple (start, stop) of seizure in patientfile

    NOTE: assumes only one seizure per file
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
    """
    Parses a patient summary file of the CHB-MIT dataset to extract
    seizure information.

    Arguments:
        summaryfile: full path to patient summary file incl. name

    Returns:
        infodict: a dictionary of seizure information
            keys: edf file names
            values: a dictionary of file-specific info 
                keys: 's', seizure indices
                values: number of seizures; tuples (start, stop)
                    of seizure onset and end samples
    """

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
                # convert from seconds to samples (fs = 256 Hz)
                start = int(lst[-2])*256
            if line.find('Seizure') != -1 and line.find('End Time') != -1:
                linsplit = line.split(' ')
                lst = [i.strip() for i in linsplit if i.strip() != '']
                # convert from seconds to samples (fs = 256 Hz)
                stop = int(lst[-2])*256
                infodict[edf][cnt] = (start, stop)
                cnt += 1
    return infodict


def butter_bandpass_filter(data, fs, lowcut=0.1, highcut=40, order = 2):
    """
    Filters input data with a Butterworth bandpass filter.
        cutoffs (default): 0.1 Hz and 40 Hz

    Arguments:
        data: input data array, channels x time
        fs: sampling frequency (in Hz)
        lowcut: lowpass filter cutoff (optional, default: 0.1 Hz)
        highcut: highpass filter cutoff (optional, default: 40 Hz)
        order: filter order (optional, default: 2)

    Returns:
        Result of filtering the input data array.
    """
    # convert to nyquist frequencies
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # digital filter specification
    i, u = signal.butter(order, [low, high], btype='bandpass', analog=False)
    y = signal.filtfilt(i, u, data, padlen=150)
    return y


def downsample(X, factor):
    """
    Downsamples the input data array by the requested factor. Downsampling
    performed by averaging across time in chunks of size "factor".

    Arguments:
        X: input data array of shape n channels x m timesteps
        factor: downsampling factor (integer)

    Returns:
        Downsampled data array, n x int(m/factor)
    """
    if factor > 1:
        X_out = np.zeros((X.shape[0], int(X.shape[1]/factor)))
        for x in range(0, X.shape[1], factor):
            X_out[:, int(x/factor)] = np.mean(X[:, x:x+factor], axis=1)
    else:
        X_out = X
    return X_out