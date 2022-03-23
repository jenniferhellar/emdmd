import os
import platform


DATASETS = ['chb-mit', 'kaggle-ieeg']

ORIG_FS = {'chb-mit': 256, 'kaggle-ieeg': 5000}

PATIENTS = {'chb-mit': ['chb01', 'chb02', 'chb04', 'chb05', 'chb06', 
						'chb07', 'chb09', 'chb10', 'chb13', 'chb14', 
						'chb16', 'chb17', 'chb18', 'chb20', 'chb22', 
						'chb23'],
			'kaggle-ieeg': ['Patient_1', 'Patient_2']}

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
							'Patient_2': {'general':[i for i in range(24)]},
							'Dog_1': {'general':kaggle_dog_ch},
							'Dog_2': {'general':kaggle_dog_ch},
							'Dog_3': {'general':kaggle_dog_ch},
							'Dog_4': {'general':kaggle_dog_ch},
							'Dog_5': {'general':kaggle_dog_ch}}
}

# downsampling factor
DS_FACTOR = {'chb-mit': 1, 'kaggle-ieeg': 10}

# averaging window size (in seconds)
W = 30

# SVD truncation parameter
R = {'chb-mit': 40, 'kaggle-ieeg': 24}

# cross-validation split (5 = 80 training/20 test split)
K = 5

# rng seed for reproducible results
SEED = 42

def get_dirs(dataset):
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
	return int(ORIG_FS[dataset]/DS_FACTOR[dataset])


def get_label_rules(fs):
	# 60 min * 60 sec/min * fs samples/sec prior to seizure
	preictal_t = 60*60*fs
	# 1 hour after seizure end
	postictal_t = 60*60*fs
	# 4 hours from any seizure
	interictal_t = 4*60*60*fs

	return preictal_t, postictal_t, interictal_t