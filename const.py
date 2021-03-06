"""
Dataset constants, key EmDMD and simulation parameters, and functions to 
obtain directory paths, effective sampling frequencies, and
preictal/interictal labelling rules.
-----------------------------------------------------------------------------

file: const.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: os, platform

-----------------------------------------------------------------------------
"""
import os
import platform


""" Dataset information """
DATASETS = ['chb-mit', 'kaggle-ieeg', 'kaggle-dog']

# sampling frequency
ORIG_FS = {'chb-mit': 256, 'kaggle-ieeg': 5000, 'kaggle-dog': 400}

# patients to analyze
PATIENTS = {'chb-mit': ['chb01', 'chb02', 'chb04', 'chb05', 'chb06', 
						'chb07', 'chb09', 'chb10', 'chb13', 'chb14', 
						'chb16', 'chb17', 'chb18', 'chb20', 'chb22', 
						'chb23'],
			'kaggle-ieeg': ['Patient_1', 'Patient_2'],
			'kaggle-dog': ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']}

# channels to use
#	some channels get shuffled in CHB-MIT files
chb_mit_normal_ch = [i for i in range(18)]
chb_mit_alt_ch = [0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11]

kaggle_dog_ch = [i for i in range(16)]
CHANNELS = {'chb-mit': {'chb01': {'general':chb_mit_normal_ch},
						'chb02': {'general':chb_mit_normal_ch},
						'chb03': {'general':chb_mit_normal_ch},
						'chb04': {'general':chb_mit_normal_ch},
						'chb05': {'general':chb_mit_normal_ch},
						'chb06': {'general':chb_mit_normal_ch},
						'chb07': {'general':chb_mit_normal_ch},
						'chb08': {'general':chb_mit_normal_ch},
						'chb09': {'general':chb_mit_normal_ch},
						'chb10': {'general':chb_mit_normal_ch},
						'chb11': {'chb11_01.edf':chb_mit_normal_ch,
								'general':chb_mit_alt_ch},
						'chb13': {'general':chb_mit_alt_ch},
						'chb14': {'general':chb_mit_alt_ch},
						'chb15': {'chb15_01.edf':chb_mit_alt_ch,
								'general':[0,1,2,3,5,6,7,8,14,15,16,17,19,20,21,22,10,11]},
						'chb16': {'general':chb_mit_alt_ch},
						'chb17': {'general':chb_mit_alt_ch},
						'chb18': {'general':chb_mit_alt_ch},
						'chb19': {'general':chb_mit_alt_ch},
						'chb20': {'general':chb_mit_alt_ch},
						'chb21': {'general':chb_mit_alt_ch},
						'chb22': {'general':chb_mit_alt_ch},
						'chb23': {'general':chb_mit_normal_ch},
						'chb24': {'general':chb_mit_normal_ch}},
			'kaggle-ieeg': {'Patient_1': {'general':[i for i in range(15)]},
							'Patient_2': {'general':[i for i in range(24)]}},
			'kaggle-dog': {'Dog_1': {'general':kaggle_dog_ch},
							'Dog_2': {'general':kaggle_dog_ch},
							'Dog_3': {'general':kaggle_dog_ch},
							'Dog_4': {'general':kaggle_dog_ch},
							'Dog_5': {'general':[i for i in range(15)]}}
}



""" EmDMD parameters """
# downsampling factor
DS_FACTOR = {'chb-mit': 1, 'kaggle-ieeg': 10, 'kaggle-dog': 1}

# averaging window size (in seconds)
W = 30

# SVD truncation parameter
# these were obtained from estimate_r.py based on keeping 90% variance
# but probably, need to be chosen based on keeping higher freq of DMD modes
R = {'chb-mit': 100, 'kaggle-ieeg': 100, 'kaggle-dog': 100}



""" Experiment/simulation parameters """
# cross-validation split (5 = 80 training/20 test split)
K = 5

# rng seed for reproducible results
SEED = 42


""" Functions """
def get_dirs(dataset):
	"""
	Depending on the system platform and input dataset, specifies
	the full paths to the data, results, and logs directories.

	Arguments:
		dataset: the dataset being analyzed should be located at
			emdmd/data/[dataset]/

	Returns:
		DATADIR: the full path to the dataset directory
		RESDIR: the full path to the results directory
			(created at emdmd/tempdata/[dataset]/ if not found)
		LOGDIR: the full path to the output logs directory
			(created at emdmd/logs/ if not found)
	"""
	mySys = platform.system()

	if mySys == 'Linux':
		usr = os.environ.get('USER')
		CODEDIR = '/home/' + usr + '/emdmd'
		if usr == 'jlh24':
			WORKDIR = '/media/large_disk/jlh24/emdmd'
		else:
			WORKDIR = CODEDIR
	elif mySys == 'Windows':
		CODEDIR =  'D:\\EmDMD\\emdmd'
		WORKDIR = 'D:\\EmDMD\\emdmd'	
	else:
		print('\nERROR: unknown operating system. Must specify CODEDIR and WORKDIR in const.py.\n')
		exit(1)

	DATADIR = os.path.join(WORKDIR, 'data', dataset)
	if not os.path.isdir(DATADIR):
		print('\n\nERROR: unable to locate data. Expected to find the dataset at ', DATADIR)
		exit(1)

	RESDIR = os.path.join(CODEDIR, 'tempdata', dataset)
	if not os.path.isdir(RESDIR):
		os.makedirs(RESDIR)

	LOGDIR = os.path.join(CODEDIR, 'logs')
	if not os.path.isdir(LOGDIR):
		os.makedirs(LOGDIR)

	return DATADIR, RESDIR, LOGDIR


def get_fs(dataset):
	"""
	Computes the effective sampling frequency of the input dataset
	after preictal/interictal segments are extracted, downsampled,
	and filtered by extract_seg.py.

	Arguments:
		dataset: the name of the dataset

	Returns:
		The integer sampling frequency of the extracted segments.
	"""
	return int(ORIG_FS[dataset]/DS_FACTOR[dataset])


def get_label_rules(fs):
	"""
	Specifies the preictal, postictal, and interictal labelling rules.

	Arguments:
		fs: original sampling frequency of the data

	Returns:
		preictal_t: the number of preictal samples prior to seizure onset
		postictal_t: no. of postictal samples after seizure end
		interictal_t: no. of samples before/after seizure to exclude
			from interictal data
	"""
	# 60 min * 60 sec/min * fs samples/sec prior to seizure
	preictal_t = 60*60*fs
	# 1 hour after seizure end
	postictal_t = 60*60*fs
	# 4 hours from any seizure
	interictal_t = 4*60*60*fs

	return preictal_t, postictal_t, interictal_t