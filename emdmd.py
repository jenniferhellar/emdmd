"""
Embedded Dynamic Mode Decomposition (EmDMD) library.

file: emdmd.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

requirements: numpy, cmath, math
"""

import numpy as np
from numpy import linalg

import cmath
import math


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
        h = math.ceil(2*m/n * 1.5)    # hn > 2m

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
    # sort and return only the positive frequencies (since symmetric)
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    return freq[int(len(freq)/2):], sort_idx


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
    freq, sort_idx = get_freq(L, fs)

    Phi = Phi[:, sort_idx]      # sort modes by frequency
    Phi = Phi[:, int(r/2):]     # select only positive frequencies
    # print(Phi.shape)            # n x r/2

    return Phi, freq


def get_power(Phi):
    """
    Computes the power of the input EmDMD mode array.

    Arguments:
        Phi: n x (r/2) EmDMD mode array.

    Returns:
        n x (r/2) power array.
    """
    # # A possibly interesting alternative 
    # # (looking at power per mode rather than per element)
    # n_modes = Phi.shape[1]
    # power = np.zeros(n_modes)
    # for mode in range(0, n_modes):
    #     power[mode] = np.sqrt(np.sum(np.abs(Phi[:,mode])**2))**2
    
    # magnitude of complex values
    power = np.abs(Phi)
    return power


def get_phase_synch(Phi):
    """
    Computes the phase synchronization of the input EmDMD mode array.

    Arguments:
        Phi: n x (r/2) EmDMD mode array.

    Returns:
        n x n phase synchronization array.
    """
    # complex phases of each element
    phase = np.zeros(Phi.shape)
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]):
            phase[i, j] = cmath.phase(Phi[i,j])
    # channel-wise Pearson correlation
    phase_corr = np.corrcoef(phase)
    # "synchronization" can be pos or neg correlation
    phase_synch =  np.abs(phase_corr)
    return phase_synch


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
        Freq: r/2 positive frequencies assoc with EmDMD modes
        Power: n x (r/2) EmDMD mode power array
        Phase_Synch: n x n EmDMD phase synchronization array
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

    # keep only the positive frequencies (symmetric)
    Freq = np.zeros(int(r/2))
    Power = np.zeros((n, int(r/2)))
    Phase_Synch = np.zeros((n, n))

    # compute DMD power/phase/freq for each sub-window of data
    for i in range(n_windows):

        # sub-window of wsub_sec
        x = X[:, i*w:(i+1)*w]

        Phi, freq = emdmd(x, fs, r, h)

        power = get_power(Phi)
        # print(power.shape)          # n x r/2

        phase_synch = get_phase_synch(Phi)       # n x n

        Freq = np.add(Freq, freq)
        Power = np.add(Power, power)
        Phase_Synch = np.add(Phase_Synch, phase_synch)

    # average over all sub-windows of the input data
    Freq = Freq/n_windows
    Power = Power/n_windows
    Phase_Synch = Phase_Synch/n_windows

    return Freq, Power, Phase_Synch


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
        An (m/(w*fs) - 1) x 2 EmDMD feature array, with the power and phase features
        in the first and second columns respectively.
    """
    w = w*fs                # window size in samples
    n_seg = X.shape[1]/w    # number of windows

    # check that we have an integer number of windows
    if int(n_seg) != n_seg:
        print('Error: get_emdmd_features() requires dim 1 of X to be a multiple of w*fs.')
        exit(0)
    else:
        n_seg = int(n_seg) 

    # taking the L2 of consecutive windows --> no feature for the first window
    l2_power = np.zeros((n_seg-1,1))
    l2_phase = np.zeros((n_seg-1,1))

    for i in range(n_seg):
        # i'th time window of data, all channels
        seg = X[:, i*w:(i+1)*w]
        freq, power, phase_synch = get_averaged_features(seg, fs, r, wsub_sec=wsub_sec, h=h)

        # compute L2 distance from the previous window's matrices
        if i > 0:
            l2_power[i-1,0] = np.sqrt(np.sum((power-last_power)**2))
            l2_phase[i-1,0] = np.sqrt(np.sum((phase_synch-last_phase_synch)**2))

        last_power = power
        last_phase_synch = phase_synch

    return np.concatenate((l2_power, l2_phase), axis=1)


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