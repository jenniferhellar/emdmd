import os
import pickle
import numpy as np
import math
from numpy.fft import fft

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import random
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn import svm
from matplotlib.patches import Rectangle
import emdmd
import util
import const

def make_meshgrid(x, y, h=.1):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx, yy, Z, **params)
	return out

PLOT_MODES			=	False 		# Fig. 2 in the paper (with chb-mit, chb01)
PLOT_EEG            =   False 		# Fig. 3 in the paper (with chb-mit, chb01)
PLOT_PLV      		=   False 		# Fig. 4 and 5 in the paper (with chb-mit, chb01)
PLOT_SPH			=	False 		# Fig. 6
PLOT_PCA			=	False 		# Fig. 7(a) chb-mit, chb01, (b) kaggle-dog, Dog_1

# some misc simulations and tests
PLOT_FEAT			=	False
PROCESS_FEATURES	=	False
PLOT_POW_TIME		=	False
CLASSIFY			=	False
CLASSIFY_KFOLDS		=	False


"""
IMPORTANT: must extract preictal/interictal segments first
"""


# dataset = 'chb-mit'
# patient = 'chb01'
# if PLOT_EEG or PLOT_PLV or PLOT_POW_TIME or PLOT_MODES or PLOT_SPH:
# 	preindices = ['0']
# 	interindices = ['0']
# else:
# 	preindices = ['0', '1', '2', '3', '4']
# 	interindices = ['0', '1', '2', '4', '5']

# dataset = 'kaggle-dog'
# patient = 'Dog_1'
# if PLOT_EEG or PLOT_PLV or PLOT_POW_TIME or PLOT_MODES or PLOT_SPH:
# 	preindices = ['0']
# 	interindices = ['0']
# else:
# 	preindices = ['0', '1', '2', '3']
# 	interindices = ['0', '1', '2', '3']

# dataset original sampling frequency
ORIG_FS = const.ORIG_FS[dataset]
# channels to include for this patient
CHANNELS = const.CHANNELS[dataset][patient]
# data directory and output directory
DATADIR, RESDIR, _ = const.get_dirs(dataset)

# downsampling factor and effective sampling frequency after downsampling
DS_FACTOR = const.DS_FACTOR[dataset]
FS = const.get_fs(dataset)
W = const.W
R = const.R[dataset]

OUTDIR = os.path.join(RESDIR, patient)
outSubDir = os.path.join(OUTDIR, 'features')

print('\n\nProcessing '+ dataset + ' record ' + patient + ' ...\n')


# Fig 2
if PLOT_MODES:
	for preidx, interidx in zip(preindices, interindices):
		f = patient + '_preictal' + preidx + '.pkl'

		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_pre = pickle.load(f)
		f.close()

		f = patient + '_interictal' + interidx + '.pkl'
		
		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_inter = pickle.load(f)
		f.close()

		if dataset == 'chb-mit':
			# 30-second window 5 min prior to seizure
			cut = (5*60 + W)*FS
			end = 5*60*FS
			x_pre = util.butter_bandpass_filter(x_pre[:, -cut:-end], fs=FS)
		else:
			# Kaggle dataset already cuts out the 5 min prior to seizure
			cut = W*FS
			end = 0
			x_pre = util.butter_bandpass_filter(x_pre[:, -cut:], fs=FS)
		
		x_inter = util.butter_bandpass_filter(x_inter[:, end:cut], fs=FS)
		n_windows = int(x_pre.shape[1]/FS)

		n = x_pre.shape[0]
		MPLV_pre = np.zeros((n,n))
		P_pre = np.zeros(5)
		MPLV_inter = np.zeros((n,n))
		P_inter = np.zeros(5)
		for i in range(1):
			# sub-window of wsub_sec
			x = x_pre[:, i*FS:(i+1)*FS]

			Phi, freq = emdmd.emdmd(x, FS, 200, None)

			power = np.abs(Phi)

			phase = np.angle(Phi) + np.pi

			fig, axes = plt.subplots(1,3,figsize=(16,4),dpi = 100)
			ax = sns.heatmap(x/np.max(x), cmap = 'coolwarm', vmin=0, vmax=1, ax = axes[0], yticklabels=False, xticklabels=False)
			if dataset == 'chb-mit':
				title = 'Preictal sEEG\n(normalized to [0,1])'
			else:
				title = 'Preictal iEEG\n(normalized to [0,1])'
			axes[0].set_title(title, fontsize=20)
			axes[0].set_ylabel('Channel', fontsize=18)
			axes[0].set_xlabel('Time', fontsize=18)
			axes[0].tick_params(left=False, bottom=False, labelsize = 18)
			ax.figure.axes[-1].tick_params(labelsize=18)

			ax = sns.heatmap(power/np.max(power), cmap = 'coolwarm', vmin=0, vmax=1, ax = axes[1], yticklabels=False, xticklabels=False)
			axes[1].set_title(r'Magnitude of $\Phi$' +'\n(normalized to [0,1])', fontsize=20)
			axes[1].set_ylabel('channel', fontsize=18)
			axes[1].set_xlabel('frequency', fontsize=18)
			axes[1].tick_params(left=False, bottom=False, labelsize = 18)
			ax.figure.axes[-1].tick_params(labelsize=18)

			ax = sns.heatmap(phase, cmap = 'coolwarm', vmin=0, vmax=2*np.pi, ax = axes[2], yticklabels=False, xticklabels=False)
			axes[2].set_title(r'Phase of $\Phi$', fontsize=20)
			axes[2].set_ylabel('channel', fontsize=18)
			axes[2].set_xlabel('frequency', fontsize=18)
			axes[2].tick_params(left=False, bottom=False, labelsize = 18)
			ax.figure.axes[-1].tick_params(labelsize=18)
			fig.tight_layout()
			plt.show()


			x = x_inter[:, i*FS:(i+1)*FS]
			Phi, freq = emdmd.emdmd(x, FS, 100, None)
			power = np.abs(Phi)
			phase = np.angle(Phi) + np.pi

			fig, axes = plt.subplots(1,3,figsize=(16,4),dpi = 100)
			ax = sns.heatmap(x/np.max(x), cmap = 'coolwarm', vmin=0, vmax=1, ax = axes[0], yticklabels=False, xticklabels=False)
			if dataset == 'chb-mit':
				title = 'Interictal sEEG\n(normalized to [0,1])'
			else:
				title = 'Interictal iEEG\n(normalized to [0,1])'
			axes[0].set_title(title, fontsize=20)
			axes[0].set_ylabel('Channel', fontsize=18)
			axes[0].set_xlabel('Time', fontsize=18)
			axes[0].tick_params(left=False, bottom=False, labelsize = 18)
			ax.figure.axes[-1].tick_params(labelsize=18)

			ax = sns.heatmap(power/np.max(power), cmap = 'coolwarm', vmin=0, vmax=1, ax = axes[1], yticklabels=False, xticklabels=False)
			axes[1].set_title(r'Magnitude of $\Phi$' + '\n(normalized to [0,1])', fontsize=20)
			axes[1].set_ylabel('channel', fontsize=18)
			axes[1].set_xlabel('frequency', fontsize=18)
			axes[1].tick_params(left=False, bottom=False, labelsize = 18)
			ax.figure.axes[-1].tick_params(labelsize=18)

			ax = sns.heatmap(phase, cmap = 'coolwarm', vmin=0, vmax=2*np.pi, ax = axes[2], yticklabels=False, xticklabels=False)
			axes[2].set_title(r'Phase of $\Phi$', fontsize=20)
			axes[2].set_ylabel('channel', fontsize=18)
			axes[2].set_xlabel('frequency', fontsize=18)
			axes[2].tick_params(left=False, bottom=False, labelsize = 18)
			ax.figure.axes[-1].tick_params(labelsize=18)
			fig.tight_layout()
			plt.show()


# Fig 3
if PLOT_EEG:
	# first 5 channels
	ch = [i for i in range(0, 5)]
	# 30-sec window 5-min prior to seizure
	cut = (5*60 + W)*FS
	end = 5*60*FS
	# convert samples to seconds
	xsec = [i/FS for i in range(cut-end)]
	for preidx, interidx in zip(preindices, interindices):
		f = patient + '_preictal' + preidx + '.pkl'

		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_pre = pickle.load(f)
		f.close()

		f = patient + '_interictal' + interidx + '.pkl'
		
		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_inter = pickle.load(f)
		f.close()
	
		seg = x_pre[:, -cut:-end]

		fig = plt.figure(figsize=(10, 6), dpi = 100)
		ax = fig.add_subplot(1,1,1) 
		# use offsets to plot multiple channels vertically
		c = -100
		for i in ch:
			plt.plot(xsec, seg[i, :] + c, lw = 2)
			c += 400
		ax.set_yticks([i*400-100 for i in range(len(ch))], ch)
		ax.tick_params(labelsize = 20)
		plt.title('Preictal sEEG, case {} (CHB-MIT)'.format(patient[-2:]), fontsize=25)
		plt.xlabel('Time (seconds)', fontsize = 20)
		plt.ylabel('Channels', fontsize = 20)
		ax.grid()
		plt.show()

		seg = x_inter[:, end:cut]

		fig = plt.figure(figsize=(10, 6), dpi = 100)
		ax = fig.add_subplot(1,1,1)
		# use offsets to plot multiple channels vertically
		c = -100
		for i in ch:
			plt.plot(xsec, seg[i, :] + c, lw = 2)
			c += 400
		# this probably isn't right but it works for channels 10-17; maybe change 10 to ch[0]?
		ax.set_yticks([i*400-100 for i in range(len(ch))], ch)
		ax.tick_params(labelsize = 20)
		plt.title('Interictal sEEG, case {} (CHB-MIT)'.format(patient[-2:]), fontsize=25)
		plt.xlabel('Time (seconds)', fontsize = 20)
		plt.ylabel('Channels', fontsize = 20)
		ax.grid()
		plt.show()


# Fig 4 and 5
if PLOT_PLV:
	for preidx, interidx in zip(preindices, interindices):
		f = patient + '_preictal' + preidx + '.pkl'

		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_pre = pickle.load(f)
		f.close()

		f = patient + '_interictal' + interidx + '.pkl'
		
		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_inter = pickle.load(f)
		f.close()

		if dataset == 'chb-mit':
			cut = (5*60 + W)*FS
			end = 5*60*FS
			x_pre = util.butter_bandpass_filter(x_pre[:, -cut:-end], fs=FS)
		else:
			cut = W*FS
			end = 0
			x_pre = util.downsample(x_pre, DS_FACTOR)
			x_pre = util.butter_bandpass_filter(x_pre[:, -cut:], fs=FS)
		
		x_inter = util.butter_bandpass_filter(x_inter[:, end:cut], fs=FS)
		n_windows = int(x_pre.shape[1]/FS)

		n = x_pre.shape[0]
		MPLV_pre = np.zeros((n,n))
		P_pre = np.zeros(5)
		MPLV_inter = np.zeros((n,n))
		P_inter = np.zeros(5)
		for i in range(n_windows):
			# sub-window of wsub_sec
			x = x_pre[:, i*FS:(i+1)*FS]
			Phi, freq = emdmd.emdmd(x, FS, 100, None)
			plv = emdmd.get_plv_matrix(Phi)
			MPLV_pre = np.add(MPLV_pre, plv)
			p = emdmd.get_subband_power(Phi, freq)
			P_pre = np.add(P_pre, p)

			x = x_inter[:, i*FS:(i+1)*FS]
			Phi, freq = emdmd.emdmd(x, FS, 100, None)
			plv = emdmd.get_plv_matrix(Phi)
			MPLV_inter = np.add(MPLV_inter, plv)
			p = emdmd.get_subband_power(Phi, freq)
			P_inter = np.add(P_inter, p)

		MPLV_pre /= n_windows
		MPLV_inter /= n_windows
		P_pre /= n_windows
		P_inter /= n_windows

		fig, axes = plt.subplots(1,1,figsize=(8,4),dpi = 100)
		ax = sns.heatmap(MPLV_pre, cmap = 'coolwarm', vmin=0, vmax=1, ax = axes, yticklabels=False, xticklabels=False)
		axes.set_title('Preictal Mean PLV', fontsize=20)
		axes.set_ylabel('Channel', fontsize=18)
		axes.set_xlabel('Channel', fontsize=18)
		axes.tick_params(left=False, bottom=False, labelsize = 18)
		ax.figure.axes[-1].tick_params(labelsize=18)
		fig.tight_layout()
		plt.show()

		fig = plt.figure(figsize=(10, 6), dpi = 100)
		ax = fig.add_subplot(1,1,1)
		xidx = [i for i in range(5)]
		xlbl = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']
		plt.plot(xidx, P_pre, 'ko-')
		ax.set_xticks(xidx, xlbl)
		ax.set_ylim(0, 0.5)
		ax.tick_params(labelsize = 18)
		plt.title('Preictal Subband Power, case {} (CHB-MIT)'.format(patient[-2:]), fontsize=25)
		plt.xlabel('Frequency Subband', fontsize = 20)
		plt.ylabel('Relative Power', fontsize = 20)
		ax.grid()
		fig.tight_layout()
		plt.show()

		fig, axes = plt.subplots(1,1,figsize=(8,4),dpi = 100)
		ax = sns.heatmap(MPLV_inter, cmap='coolwarm', vmin=0, vmax=1, ax = axes, yticklabels=False, xticklabels=False)
		axes.set_title('Interictal Mean PLV', fontsize=20)
		axes.set_ylabel('Channel', fontsize=18)
		axes.set_xlabel('Channel', fontsize=18)
		axes.tick_params(left=False, bottom=False, labelsize = 18)
		ax.figure.axes[-1].tick_params(labelsize=18)
		fig.tight_layout()
		plt.show()

		fig = plt.figure(figsize=(10, 6), dpi = 100)
		ax = fig.add_subplot(1,1,1)
		xidx = [i for i in range(5)]
		xlbl = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']
		plt.plot(xidx, P_inter, 'ko-')
		ax.set_xticks(xidx, xlbl)
		ax.set_ylim(0, 0.5)
		ax.tick_params(labelsize = 18)
		plt.title('Interictal Subband Power, case {} (CHB-MIT)'.format(patient[-2:]), fontsize=25)
		plt.xlabel('Frequency Subband', fontsize = 20)
		plt.ylabel('Relative Power', fontsize = 20)
		ax.grid()
		fig.tight_layout()
		plt.show()


# Fig 6
if PLOT_SPH:
	# 40 min prior to 5 min after seizure onset
	cut = 40*60*FS
	end = 5*60*FS

	sop2 = 35
	sop1 = 15
	sph = 5
	# convert samples to seconds and set t=0 to onset
	xsec = [(i-cut)/(60*FS) for i in range(cut+end)]
	fname_preictal = 'chb01_03.edf'

	fpath = os.path.join(DATADIR, patient)
	summarypath = os.path.join(fpath, patient+'-summary.txt')
	(seizure_start, seizure_end) = util.get_file_summary(fname_preictal, summarypath)
	sflag = seizure_start*FS

	x_all = util.read_edf(os.path.join(fpath, fname_preictal))

	seg = x_all[0, (sflag-cut):(sflag+end)]
	seg = util.butter_bandpass_filter(seg, fs=FS)

	fig = plt.figure(figsize=(10, 6), dpi = 100)
	ax = fig.add_subplot(1,1,1) 
	plt.plot(xsec, seg, c='k', lw = 2)
	ax.axvline(x = 0, label = 'Seizure starts', c = 'r')
	# seizure prediction horizon
	ax.axvline(x = -sph, c = 'b')
	plt.annotate('', xy=(-sph, 400), xycoords='data', xytext=(0, 400), textcoords='data',
   		arrowprops={'arrowstyle': '<->', 'color':'b'})
	plt.annotate('SPH', xy=(-sph+1, 410), xycoords='data', textcoords='data', c='b', fontsize = 16)
	ax.tick_params(labelsize = 16)
	# seizure occurrence period of 10 min
	ax.axvline(x = -sop1, c = 'g')
	plt.annotate('', xy=(-sop1, 400), xycoords='data', xytext=(-sph, 400), textcoords='data',
   		arrowprops={'arrowstyle': '<->', 'color':'g'})
	plt.annotate(r'SOP$_1$', xy=(-sop1+3, 410), xycoords='data', textcoords='data', c='g', fontsize = 16)
	# seizure occurrence period of 30 min
	ax.axvline(x = -sop2, c = 'purple')
	plt.annotate('', xy=(-sop2, 300), xycoords='data', xytext=(-sph, 300), textcoords='data',
   		arrowprops={'arrowstyle': '<->', 'color':'purple'})
	plt.annotate(r'SOP$_2$', xy=(-sop2+6, 310), xycoords='data', textcoords='data', c='purple', fontsize = 16)
	# plt.title(''.format(patient[-2:]), fontsize=25)
	plt.xlabel('Time (minutes)', fontsize = 18)
	plt.ylabel('Voltage (single-channel)', fontsize = 18)
	plt.legend(fontsize = 16, loc='lower right')
	ax.grid()
	plt.show()


# Fig 7
if PLOT_PCA:
	first = True
	for preidx in preindices:
		file = os.path.join(outSubDir, patient + '_preictal' + preidx + '_features.pkl')
		f = open(file, 'rb')
		features = pickle.load(f)
		f.close()

		# 15 to 5 min prior to seizure
		cut = int(15*60/W)
		end = int(5*60/W)

		if dataset == 'chb-mit':
			# 15-5 min before seizure onset
			features = features[-cut:end, :]

		if first:
			pre_features = features
			first = False
		else:
			pre_features = np.concatenate((pre_features, features), axis=0)

	first = True
	for interidx in interindices:
		file = os.path.join(outSubDir, patient + '_interictal' + interidx + '_features.pkl')
		f = open(file, 'rb')
		features = pickle.load(f)
		f.close()

		cut = int(10*60/W)
		# 10 min before end
		features = features[-cut:, :]

		if first:
			inter_features = features
			first = False
		else:
			inter_features = np.concatenate((inter_features, features), axis=0)

	pre_n = pre_features.shape[0]
	inter_n = inter_features.shape[0]

	X = np.concatenate((pre_features, inter_features), axis=0)
	print('X:\t', X.shape)
	y = np.concatenate((np.ones(pre_n), np.zeros(inter_n)), axis=0)
	print('y:\t', y.shape)

	K = const.K
	SEED = const.SEED

	random.seed(a=SEED, version=2)

	kf = KFold(n_splits=K, random_state=SEED, shuffle=True)
	print('\n')
	for tr_i, tst_i in kf.split(X):
		X_tr = X[tr_i, :]
		X_tst = X[tst_i, :]
		y_tr = y[tr_i]
		y_tst = y[tst_i]

		scaler = StandardScaler()
		scaler.fit(X_tr)
		X_tr = scaler.transform(X_tr)

		selector = SelectFromModel(estimator=LogisticRegression()).fit(X_tr, y_tr)
		feat = selector.get_support()
		X_tr = X_tr[:, feat]
		X_tst = scaler.transform(X_tst)[:, feat]

		pca = PCA(n_components=2)
		pca.fit(X_tr)

		X_tr = pca.transform(X_tr)
		pca_X = pca.transform(scaler.transform(X)[:, feat])
		
		classifier = svm.SVC(kernel='linear')
		# classifier = LogisticRegression()
		clf = classifier.fit(X_tr, y_tr)

		fig, ax = plt.subplots(figsize=(8, 6), dpi = 100)
		# Set-up grid for plotting.
		X0, X1 = pca_X[:, 0], pca_X[:, 1]
		xx, yy = make_meshgrid(X0, X1)

		plot_contours(ax, clf, xx, yy, cmap=plt.cm.RdYlGn, alpha=0.5)
		pre_idx = np.where(y == 0)[0]
		inter_idx = np.where(y == 1)[0]
		ax.scatter(pca_X[pre_idx,0], pca_X[pre_idx,1], c='r', label='preictal', s=50)
		ax.scatter(pca_X[inter_idx,0], pca_X[inter_idx,1], c='g', label='interictal', s=50)
		ax.set_ylabel('Principle Component 2', fontsize = 20)
		ax.set_xlabel('Principle Component 1', fontsize = 20)
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title('Decision surface of Linear SVM', fontsize=25)
		ax.legend(fontsize = 18)
		plt.show()


# plot subset of features over time
if PLOT_FEAT:
	cut = 40*60*FS
	end = 5*60*FS

	sop2 = 35
	sop1 = 15
	sph = 5
	# convert samples to seconds
	xsec = [(i-cut)/(60*FS) for i in range(cut+end)]
	fname_preictal = 'chb01_03.edf'

	fpath = os.path.join(DATADIR, patient)
	summarypath = os.path.join(fpath, patient+'-summary.txt')
	(seizure_start, seizure_end) = util.get_file_summary(fname_preictal, summarypath)
	sflag = seizure_start*FS

	x_all = util.read_edf(os.path.join(fpath, fname_preictal))

	seg = x_all[0:18, (sflag-cut):(sflag+end)]
	seg = util.butter_bandpass_filter(seg, fs=FS)
	features = emdmd.get_emdmd_features(seg, fs=FS, w=W, r=R)

	# first 20 features
	p = features[:, :20]
	ch = [i for i in range(p.shape[1])]
	xmin = [(i - p.shape[0] + 1)*W/60 for i in range(p.shape[0])]

	fig = plt.figure(figsize=(10, 6), dpi = 100)
	ax = fig.add_subplot(1,1,1) 

	ylbl = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$M_1$', r'$M_2$', r'$M_3$']
	ylbl = ylbl + [i for i in range(p.shape[1]-8)]
	# use offsets to plot multiple channels vertically
	c = 0
	for i in ch:
		plt.plot(xmin, p[:, i] + c, 'o-')
		c += .5
	# this probably isn't right but it works for channels 10-17; maybe change 10 to ch[0]?
	ax.set_yticks([i*.5 for i in range(len(ch))], ylbl)
	ax.tick_params(labelsize = 20)
	plt.title('Preictal Features, case {} (CHB-MIT)'.format(patient[-2:]), fontsize=25)
	plt.xlabel('Time (min)', fontsize = 20)
	plt.ylabel('Features', fontsize = 20)
	plt.ylim([-0.5, (len(ch)+1)*0.5])
	ax.grid()

	plt.show()


# extract and save features from some segments
if PROCESS_FEATURES:
	R = const.R[dataset]
	cut = 15*60*FS
	for preidx, interidx in zip(preindices, interindices):
		f = patient + '_preictal' + preidx + '.pkl'

		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_pre = pickle.load(f)
		f.close()
		
		pre_features = emdmd.get_emdmd_features(x_pre[:, -cut:], fs=FS, w=W, r=R)
		print(preidx, pre_features.shape)

		output = open(os.path.join(outSubDir, patient + '_preictal' + preidx + '_features.pkl'), 'wb')
		pickle.dump(pre_features, output)
		output.close()

		f = patient + '_interictal' + interidx + '.pkl'
		
		file = os.path.join(OUTDIR, f)
		f = open(file, 'rb')
		x_inter = pickle.load(f)
		f.close()

		inter_features = emdmd.get_emdmd_features(x_inter[:, -cut:], fs=FS, w=W, r=R)
		print(interidx, inter_features.shape)

		output = open(os.path.join(outSubDir, patient + '_interictal' + interidx + '_features.pkl'), 'wb')
		pickle.dump(inter_features, output)
		output.close()


# classify without train/test splits (overestimation)
if CLASSIFY:
	first = True
	for preidx in preindices:
		file = os.path.join(outSubDir, patient + '_preictal' + preidx + '_features.pkl')
		f = open(file, 'rb')
		features = pickle.load(f)
		f.close()

		# 10-5 min before seizure onset
		cut = int(15*60/W)
		end = int(5*60/W)
		features = features[-cut:end, :]

		if first:
			pre_features = features
			first = False
		else:
			pre_features = np.concatenate((pre_features, features), axis=0)

	first = True
	for interidx in interindices:
		file = os.path.join(outSubDir, patient + '_interictal' + interidx + '_features.pkl')
		f = open(file, 'rb')
		features = pickle.load(f)
		f.close()

		if first:
			inter_features = features
			first = False
		else:
			inter_features = np.concatenate((inter_features, features), axis=0)

	pre_n = pre_features.shape[0]
	inter_n = inter_features.shape[0]

	X = np.concatenate((pre_features, inter_features), axis=0)
	print('X:\t', X.shape)
	y = np.concatenate((np.ones(pre_n), np.zeros(inter_n)), axis=0)
	print('y:\t', y.shape)

	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)

	selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
	feat = selector.get_support()
	print(list(feat))
	X = X[:, feat]

	classifier = svm.SVC(kernel='linear')
	clf = classifier.fit(X, y)

	y_pred = clf.predict(X)
	confusion_mat = confusion_matrix(y, y_pred)
	TP = confusion_mat[0][0]
	FP = confusion_mat[0][1]
	FN = confusion_mat[1][0]
	TN = confusion_mat[1][1]

	acc = (TN + TP) / (TN + FP + FN + TP)
	sen = TP / (TP + FN)
	spec = TN / (TN + FP)

	print('acc: ', acc, '\tsen: ', sen, '\tspec: ', spec)


# classify with cv
if CLASSIFY_KFOLDS:
	first = True
	for preidx in preindices:
		file = os.path.join(outSubDir, patient + '_preictal' + preidx + '_features.pkl')
		f = open(file, 'rb')
		features = pickle.load(f)
		f.close()

		cut = int(15*60/W)
		end = int(5*60/W)
		if dataset == 'chb-mit':
			# 10-5 min before seizure onset
			features = features[-cut:end, :]

		if first:
			pre_features = features
			first = False
		else:
			pre_features = np.concatenate((pre_features, features), axis=0)

	first = True
	for interidx in interindices:
		file = os.path.join(outSubDir, patient + '_interictal' + interidx + '_features.pkl')
		f = open(file, 'rb')
		features = pickle.load(f)
		f.close()

		if first:
			inter_features = features
			first = False
		else:
			inter_features = np.concatenate((inter_features, features), axis=0)

	pre_n = pre_features.shape[0]
	inter_n = inter_features.shape[0]

	X = np.concatenate((pre_features, inter_features), axis=0)
	print('X:\t', X.shape)
	y = np.concatenate((np.ones(pre_n), np.zeros(inter_n)), axis=0)
	print('y:\t', y.shape)

	K = const.K
	SEED = const.SEED

	random.seed(a=SEED, version=2)

	kf = KFold(n_splits=K, random_state=SEED, shuffle=True)
	Accuracy = 0
	Sensitivity = 0
	Specificity = 0
	print('\n')
	for tr_i, tst_i in kf.split(X):
		X_tr = X[tr_i, :]
		X_tst = X[tst_i, :]
		y_tr = y[tr_i]
		y_tst = y[tst_i]

		scaler = StandardScaler()
		scaler.fit(X_tr)
		X_tr = scaler.transform(X_tr)

		selector = SelectFromModel(estimator=LogisticRegression()).fit(X_tr, y_tr)
		feat = selector.get_support()
		X_tr = X_tr[:, feat]
		
		classifier = svm.SVC(kernel='linear')
		clf = classifier.fit(X_tr, y_tr)

		X_tst = scaler.transform(X_tst)[:, feat]
		y_pred = clf.predict(X_tst)
		confusion_mat = confusion_matrix(y_tst, y_pred)
		TP = confusion_mat[0][0]
		FP = confusion_mat[0][1]
		FN = confusion_mat[1][0]
		TN = confusion_mat[1][1]

		acc = (TN + TP) / (TN + FP + FN + TP)
		sen = TP / (TP + FN)
		spec = TN / (TN + FP)

		print('acc: ', acc, '\tsen: ', sen, '\tspec: ', spec)

		Accuracy += acc
		Sensitivity += sen
		Specificity += spec

	print('\n\nMean CV Metrics')
	print('accuracy:\t', Accuracy/K)
	print('sensitivity:\t', Sensitivity/K)
	print('specificity:\t', Specificity/K)
