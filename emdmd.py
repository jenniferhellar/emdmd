import numpy as np
from numpy import linalg

import cmath
import math


def get_hankel_matrices(X, h=None):
    n, m = X.shape
    
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
    # singular value decomposition
    U, s, Vt = linalg.svd(X, full_matrices=False)
    # S = np.diag(s)              # (m-h) x (m-h)

    U_r = U[:, :r]                  # nh x r
    S_r = np.diag(s[:r])            # r x r
    V_r = np.transpose(Vt[:r, :])   # (m-h) x r

    return U_r, S_r, V_r


def dmd(U, S, V, Xh_prime):
    A_tilde = np.dot(V, linalg.inv(S))
    A_tilde = np.dot(Xh_prime, A_tilde)
    A_tilde = np.dot(np.transpose(U), A_tilde)

    # A_tilde = np.dot(np.sqrt(linalg.inv(S)), np.dot(A_tilde, np.sqrt(S)))
    eig_val, W = linalg.eig(A_tilde)
    L = np.diag(eig_val)    # r x r eigendecomposition matrices

    # # PhiH = np.dot(np.sqrt(S), W)
    # PhiH = np.dot(linalg.inv(S), W)
    # PhiH = np.dot(V, PhiH)
    # PhiH = np.dot(Xh_prime, PhiH)     # nh x r (EmDMD modes)

    PhiH = np.dot(U, W)

    return PhiH, L


def get_freq(L, fs):
    delta_t = 1/fs
    freq = np.zeros(L.shape[0])
    for i in range(L.shape[0]):
        theta = cmath.phase(L[i, i])
        freq[i] = theta/(2*np.pi*delta_t)
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    return freq[int(len(freq)/2):], sort_idx


def emdmd(x, fs, r, h=None):
    # Hankel matrices: nh x (w-h)
    Xh, Xh_prime, h = get_hankel_matrices(x, h=h)
    # print('Hankel matrix shapes:\n\t', Xh.shape, Xh_prime.shape)

    U, S, V = get_trunc_svd(Xh, r=r)

    # EmDMD modes PhiH, nh x r matrix
    # DMD eigenvalues L, r x r matrix
    PhiH, L = dmd(U, S, V, Xh_prime)

    # (undoing the time embedding)
    n = int(PhiH.shape[0]/h)
    Phi = PhiH[:n, :]
    # print(Phi.shape)            # n x r

    # Phi = Phi/np.sqrt(np.sum(np.abs(Phi)**2))

    freq, sort_idx = get_freq(L, fs)

    Phi = Phi[:, sort_idx]      # sort by frequency
    Phi = Phi[:, int(r/2):]     # select only positive frequencies
    # print(Phi.shape)            # n x r/2

    return Phi, freq


def get_power(Phi):
    # n_modes = Phi.shape[1]
    # power = np.zeros(n_modes)
    # for mode in range(0, n_modes):
    #     power[mode] = np.sqrt(np.sum(np.abs(Phi[:,mode])**2))**2
    power = np.abs(Phi)
    return power


def get_phase_synch(Phi):
    phase = np.zeros(Phi.shape)
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]):
            phase[i, j] = cmath.phase(Phi[i,j])
    phase_corr = np.corrcoef(phase)
    phase_synch =  np.abs(phase_corr)
    return phase_synch


def get_averaged_features(X, fs, r, wsub_sec=1, h=None):
    """
    X: input window of n channels x m timesteps
    h: Hankel embedding parameter
    r: SVD truncation parameter
    wsub_sec: sub-window size (in seconds) to compute dmd power/phase
    fs: sampling rate (in Hz) of input data 
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


def get_emdmd_features(X, fs, r, w_sec=30, wsub_sec=1, h=None):
    w = (w_sec*fs)
    n_seg = X.shape[1]/w
    if int(n_seg) != n_seg:
        print('Error: get_emdmd_features() requires dim 1 of X to be a multiple of w_sec*fs.')
        exit(0)
    else:
        n_seg = int(n_seg) 

    l2_power = np.zeros((n_seg-1,1))
    l2_phase = np.zeros((n_seg-1,1))

    for i in range(n_seg):
        seg = X[:, i*w:(i+1)*w]
        freq, power, phase_synch = get_averaged_features(seg, fs, r, wsub_sec=wsub_sec, h=h)

        if i > 0:
            l2_power[i-1,0] = np.sqrt(np.sum((power-last_power)**2))
            l2_phase[i-1,0] = np.sqrt(np.sum((phase_synch-last_phase_synch)**2))
        last_power = power
        last_phase_synch = phase_synch

    return np.concatenate((l2_power, l2_phase), axis=1)


def get_r(X, p=0.90):
    # singular value decomposition
    W, s, Ut = linalg.svd(X, full_matrices=False)

    # default to keeping enough to cover p variance in data
    var = s**2/np.sum(s**2)
    r = next(x for x, val in enumerate(np.cumsum(var))
                              if val > p)

    return r


def estimate_r(X, fs, wsub_sec=1, h=None, p=0.90):
    """
    X: input window of n channels x m timesteps
    h: Hankel embedding parameter
    r: SVD truncation parameter
    wsub_sec: sub-window size (in seconds) to compute dmd power/phase
    fs: sampling rate (in Hz) of input data 
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