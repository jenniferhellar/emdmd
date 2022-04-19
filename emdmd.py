"""
Embedded Dynamic Mode Decomposition (EmDMD) library.

file: emdmd.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

requirements: numpy, cmath, math, itertools
"""

import numpy as np
from numpy import linalg

import cmath
import math
import itertools

# local script
import util


def get_hankel_matrices(X, h=None):
    """
    Computes the Hankel time embedding augmented matrices for the
    input data array.

    Arguments:
        X: input data array, n channels x m timesteps
        h: Hankel embedding parameter (optional)

    Returns:
        Xh, Xh_prime: the nh x (m-h) augmented matrices
        h: the embedding parameter used
    """
    n, m = X.shape

    # if parameter not supplied, use this general rule
    if h == None:
        h = math.ceil(3*m/n)    # hn > 2m

    # print(n, m, h)
    H = X[:, 0:(m - h + 1)]     # embedding on the data
    for i in range(1, h):
        aug = X[:, i:(i + m - h + 1)]
        H = np.concatenate((H, aug), axis=0)
    Xh = H[:, :-1]              # nh x (m-h)
    Xh_prime = H[:, 1:]         # time-shifted by one step
    return Xh, Xh_prime, h


def get_trunc_svd(X, r):
    """
    Computes the truncated singular value decomposition of the
    input data array.

    Arguments:
        X: input data array
        r: truncation parameter (see optimize_r.py)

    Returns:
        U, S, V matrices
    """
    # singular value decomposition
    U, s, Vt = linalg.svd(X, full_matrices=False)
    # S = np.diag(s)              # (m-h) x (m-h)

    # truncate the matrices
    U_r = U[:, :r]                  # nh x r
    S_r = np.diag(s[:r])            # r x r
    V_r = np.transpose(Vt[:r, :])   # (m-h) x r

    return U_r, S_r, V_r


def dmd(U, S, V, Xh_prime):
    """
    Computes the dynamic mode decomposition w.r.t. the input SVD of
    the data matrix and w.r.t. the data matrix one step in the future.

    Arguments:
        U, S, V: the SVD decomposition of the first data matrix
            U: nh x r left singular vectors
            S: r x r singular value diagonal array
            V: (m-h) x r right singular vectors
        Xh_prime: the data matrix one step in the future

    Returns:
        PhiH: the DMD mode array, nh "channels" x r modes
        L: the DMD eigenvalue array, r x r
    """
    # estimate the operator that transforms X to X_prime
    A_tilde = np.dot(V, linalg.inv(S))
    A_tilde = np.dot(Xh_prime, A_tilde)
    A_tilde = np.dot(np.transpose(U), A_tilde)

    # eigendecomposition of A
    eig_val, W = linalg.eig(A_tilde)
    L = np.diag(eig_val)    # r x r eigendecomposition matrices

    # # alternative DMD method (see Brunton et al 2016)
    # PhiH = np.dot(linalg.inv(S), W)
    # PhiH = np.dot(V, PhiH)
    # PhiH = np.dot(Xh_prime, PhiH)     # nh x r (EmDMD modes)

    # (see Tu et al 2013, equn 2.5)
    PhiH = np.dot(U, W)

    return PhiH, L


def get_freq(L, fs):
    """
    Computes the EmDMD frequencies from the eigenvalue matrix.

    Arguments:
        L: DMD eigenvalue matrix, r x r
        fs: sampling frequency of the data (in Hz)

    Returns:
        freq: sorted frequencies associated with the EmDMD modes
        sort_idx: the indices used to sort freq (needed to sort
            the EmDMD modes by frequency)
    """
    delta_t = 1/fs
    freq = np.zeros(L.shape[0])
    # foreach eigenvalue
    for i in range(L.shape[0]):
        # complex phase
        theta = cmath.phase(L[i, i])
        # corresponding frequency
        freq[i] = theta/(2*np.pi*delta_t)
    # sort by frequency
    keep_modes = np.argsort(freq)
    freq = freq[keep_modes]
    # return only the positive frequencies (since symmetric)
    low_cut = next(i for i, f in enumerate(freq) if f > 0)
    # ignore all frequencies above 60 Hz
    if freq[-1] > 60:
        high_cut = next(i for i, f in enumerate(freq) if f > 60)
    else:
        high_cut = len(freq)

    freq = freq[low_cut:high_cut]
    keep_modes = keep_modes[low_cut:high_cut]
    return freq, keep_modes


def emdmd(x, fs, r, h=None):
    """
    Computes the time-emdedded dynamic mode decomposition of the input
    data array and extracts the modes corresponding to positive frequencies.

    Arguments:
        x: input data array, n channels x m timesteps
        fs: sampling frequency of input data (in Hz)
        r: SVD truncation parameter (see estimate_r.py)
        h: Hankel embedding parameter (optional)

    Returns:
        Phi: n x (r/2) EmDMD mode array (pos freq only)
        freq: (r/2) positive frequencies assoc. with Phi modes
    """
    # Hankel matrices: nh x (m-h)
    Xh, Xh_prime, h = get_hankel_matrices(x, h=h)
    # print('Hankel matrix shapes:\n\t', Xh.shape, Xh_prime.shape)

    # truncated singular value decomposition
    U, S, V = get_trunc_svd(Xh, r=r)

    # EmDMD modes PhiH, nh x r matrix
    #   modes: each column of PhiH
    # DMD eigenvalues L, r x r matrix
    PhiH, L = dmd(U, S, V, Xh_prime)

    # (undoing the time embedding)
    n = int(PhiH.shape[0]/h)
    Phi = PhiH[:n, :]
    # print(Phi.shape)            # n x r

    # compute the sorted frequencies and assoc. sorting index
    freq, keep_idx = get_freq(L, fs)

    Phi = Phi[:, keep_idx]      # sort modes by frequency and keep pos

    return Phi, freq


def get_power(Phi):
    """
    Computes the power of the input EmDMD mode array.

    Arguments:
        Phi: n x m EmDMD mode array.

    Returns:
        1 x m power array.
    """
    # power per mode
    n_modes = Phi.shape[1]
    power = np.zeros(n_modes)
    for mode in range(0, n_modes):
        power[mode] = np.sqrt(np.sum(np.abs(Phi[:,mode])**2))**2
    return power


def get_subband_power(Phi, freq):
    # freq upper bounds, in Hz
    delta_f = 4     # 0-4Hz
    theta_f = 8     # 4-8Hz
    alpha_f = 12    # 8-12Hz
    beta_f = 30     # 12-30Hz
    gamma_f = 60    # 30-60Hz

    Power = get_power(Phi)

    subband_p = np.zeros(5)
    for i in range(freq.shape[0]):
        f = freq[i]
        if f <= delta_f:
            subband_p[0] += Power[i]
        elif f <= theta_f:
            subband_p[1] += Power[i]
        elif f <= alpha_f:
            subband_p[2] += Power[i]
        elif f <= beta_f:
            subband_p[3] += Power[i]
        elif f <= gamma_f:
            subband_p[4] += Power[i]

    subband_p = subband_p / np.sum(subband_p)

    return subband_p



def get_moments(Phi, freq):
    m = len(freq)
    M = np.zeros(3)

    Power = get_power(Phi)
    for j in range(M.shape[0]):
        for i in range(m):
            f = freq[i]
            p = Power[i]
            M[j] += (f ** j)*p

    return M


def get_plv(Phi):
    """
    Computes the channel-wise phase synchronization of the input 
    EmDMD mode array as the phase locking values for a given pair 
    of channels, averaged across modes/frequencies.

    Arguments:
        Phi: n x m EmDMD mode array.

    Returns:
        1 x n*(n-1)/2 phase synchronization array.
    """
    # complex phases of each element
    phase = np.zeros(Phi.shape)
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]):
            phase[i, j] = cmath.phase(Phi[i,j])
    # unique pairs of channels
    ch_pairs = list(itertools.combinations([i for i in range(Phi.shape[0])], 2))
    plv = np.zeros(len(ch_pairs))
    pair_i = 0
    # for each channel pair
    for (i,j) in ch_pairs:
        theta1 = phase[i, :]    # phases of channel i
        theta2 = phase[j, :]    # phases of channel j
        # element-wise phase differences
        complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
        # average across modes/frequencies
        plv[pair_i] = np.abs(np.sum(complex_phase_diff))/len(theta1)
        pair_i += 1
    return plv


def get_plv_matrix(Phi):
    """
    Computes the channel-wise phase synchronization of the input 
    EmDMD mode array as the phase locking values for a given pair 
    of channels, averaged across modes/frequencies.

    Arguments:
        Phi: n x m EmDMD mode array.

    Returns:
        1 x n*(n-1)/2 phase synchronization array.
    """
    # complex phases of each element
    phase = np.zeros(Phi.shape)
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]):
            phase[i, j] = cmath.phase(Phi[i,j])
    n = Phi.shape[0]
    plv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            theta1 = phase[i, :]    # phases of channel i
            theta2 = phase[j, :]    # phases of channel j
            # element-wise phase differences
            complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
            # average across modes/frequencies
            plv[i, j] = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv


def get_averaged_features(X, fs, r, wsub_sec=1, h=None):
    """
    Computes the EmDMD power/phase synchronization matrices over each
    sub-window of the input data and returns the averaged results.

    Arguments:
        X: input window of n channels x m timesteps
        fs: sampling rate of input data (in Hz)
        r: SVD truncation parameter (see estimate_r.py)
        wsub_sec: sub-window size (in seconds) to compute EmDMD power/phase matrices
        h: Hankel embedding parameter
    
    Returns:
    """
    n = X.shape[0]
    m = X.shape[1]
    w = wsub_sec*fs
    n_windows = m/w
    if int(n_windows) != n_windows:
        print('Error: get_averaged_features() requires dim 1 of X to be a multiple of w_sec*fs.')
        exit(0)
    else:
        n_windows = int(n_windows)

    # filter from 0.1 to 40 Hz
    X = util.butter_bandpass_filter(X, fs=fs)

    Subband_Power = np.zeros((1, 5))
    Moments = np.zeros((1, 3))
    PLV = np.zeros((1, int(n*(n-1)/2)))
    Freq_Max = 0

    # compute DMD power/phase/freq for each sub-window of data
    for i in range(n_windows):

        # sub-window of wsub_sec
        x = X[:, i*w:(i+1)*w]

        Phi, freq = emdmd(x, fs, r, h)
        Freq_Max += freq[-1]

        subband_power = get_subband_power(Phi, freq)
        moments = get_moments(Phi, freq)
        plv = get_plv(Phi)

        Subband_Power = np.add(Subband_Power, subband_power)
        Moments = np.add(Moments, moments)
        PLV = np.add(PLV, plv)

    # average over all sub-windows of the input data
    Subband_Power /= n_windows
    Moments /= n_windows
    PLV /= n_windows
    Freq_Max /= n_windows

    return Subband_Power, Moments, PLV, Freq_Max


def get_emdmd_features(X, fs, w, r, wsub_sec=1, h=None):
    """
    Computes the EmDMD features for input data X.

    Arguments:
        X: input data array of n channels x m timesteps
        fs: sampling frequency of X (in Hz)
        w: window size (in seconds) over which to average the EmDMD modes/phases
            Note: m (time dim of X) must be a multiple of w*fs
        r: SVD truncation parameter (see estimate_r.py)
        wsub_sec: sub-window size in seconds to compute an EmDMD mode/phase pair (optional)
        h: Hankel embedding parameter (optional)

    Returns:
    """
    w = w*fs                # window size in samples
    n_seg = X.shape[1]/w    # number of windows

    # check that we have an integer number of windows
    if int(n_seg) != n_seg:
        print('Error: get_emdmd_features() requires dim 1 of X to be a multiple of w*fs.')
        exit(0)
    else:
        n_seg = int(n_seg) 

    n = X.shape[0]
    subband = np.zeros((n_seg, 5))
    moments = np.zeros((n_seg, 3))
    plv = np.zeros((n_seg, int(n*(n-1)/2)))

    for i in range(n_seg):
        # i'th time window of data, all channels
        seg = X[:, i*w:(i+1)*w]

        subband_power, m, ch_plv, _ = get_averaged_features(seg, fs, r, wsub_sec=wsub_sec, h=h)

        subband[i] = subband_power
        moments[i] = m
        plv[i] = ch_plv

    return np.concatenate((subband, moments, plv), axis=1)


def get_fmax(X, fs, w, r, wsub_sec=1, h=None):
    """
    Computes the EmDMD features for input data X.

    Arguments:
        X: input data array of n channels x m timesteps
        fs: sampling frequency of X (in Hz)
        w: window size (in seconds) over which to average the EmDMD modes/phases
            Note: m (time dim of X) must be a multiple of w*fs
        r: SVD truncation parameter (see estimate_r.py)
        wsub_sec: sub-window size in seconds to compute an EmDMD mode/phase pair (optional)
        h: Hankel embedding parameter (optional)

    Returns:
    """
    w = w*fs                # window size in samples
    n_seg = X.shape[1]/w    # number of windows

    # check that we have an integer number of windows
    if int(n_seg) != n_seg:
        print('Error: get_fmax() requires dim 1 of X to be a multiple of w*fs.')
        exit(0)
    else:
        n_seg = int(n_seg) 

    fmax = np.zeros((n_seg, 1))

    for i in range(n_seg):
        # i'th time window of data, all channels
        seg = X[:, i*w:(i+1)*w]

        _, _, _, f = get_averaged_features(seg, fs, r, wsub_sec=wsub_sec, h=h)

        fmax[i] = f

    print(np.max(fmax), np.mean(fmax), np.std(fmax), np.mean(fmax) + np.std(fmax))


def get_r(X, p=0.90):
    """
    Computes the SVD of input data X and computes the number of singular
    values that contain p fraction of overall variance.

    Arguments:
        X: input array of n channels x m timesteps
        p: fraction of variance to keep (default: 0.9)

    Returns:
        r: the integer number of singular values to keep during truncation
    """
    # singular value decomposition
    _, s, _ = linalg.svd(X, full_matrices=False)

    # keep enough to cover p variance in data
    var = s**2/np.sum(s**2)
    r = next(x for x, val in enumerate(np.cumsum(var))
                              if val > p)
    return r


def estimate_r(X, fs, wsub_sec=1, h=None, p=0.90):
    """
    Computes the SVD "r" truncation parameter needed to keep p variance
    in each sub-window of wsub_sec of the input data.

    Arguments:
        X: input window of n channels x m timesteps
        fs: sampling rate (in Hz) of input data
        wsub_sec: sub-window size in seconds (default: 1)
        h: Hankel embedding parameter (optional)
        p: fraction of variance to keep (default: 0.9)

    Returns:
        A 1D array of r values, one per window.
    """
    n = X.shape[0]
    m = X.shape[1]
    w = wsub_sec*fs

    n_windows = m/w
    if int(n_windows) != n_windows:
        print('Error: estimate_r() requires dim 1 of X to be a multiple of w_sec*fs.')
        exit(0)
    else:
        n_windows = int(n_windows)

    r_all = np.zeros(n_windows)

    for i in range(n_windows):

        # sub-window of wsub_sec
        x = X[:, i*w:(i+1)*w]

        # Hankel matrices: nh x (w-h)
        Xh, Xh_prime, h = get_hankel_matrices(x, h=h)

        r_all[i] = get_r(Xh, p=p)

    return r_all