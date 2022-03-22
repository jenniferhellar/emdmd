import os
import platform


DATASETS = ['chb-mit', 'kaggle-ieeg']


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
	if dataset == 'chb-mit':
		return 256
	elif dataset == 'kaggle-ieeg':
		return 5000


def get_patients(dataset):
	if dataset == 'chb-mit':
		PATIENTS = ['chb01', 'chb02', 'chb04', 'chb05', 'chb06', 'chb07', 'chb09',
				'chb10', 'chb13', 'chb14', 'chb16', 'chb17', 'chb18', 'chb20',
				'chb22', 'chb23']
	elif dataset == 'kaggle-ieeg':
		PATIENTS = ['Patient_1', 'Patient_2']

	return PATIENTS


def get_channels(dataset):
	if dataset == 'chb-mit':
		normal_ch = [i for i in range(18)]
		alt_ch = [0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11]
		CHANNELS = {
			'chb01': {'general':normal_ch},
			'chb02': {'general':normal_ch},
			'chb03': {'general':normal_ch},
			'chb04': {'general':normal_ch},
			'chb05': {'general':normal_ch},
			'chb06': {'general':normal_ch},
			'chb07': {'general':normal_ch},
			'chb08': {'general':normal_ch},
			'chb09': {'general':normal_ch},
			'chb10': {'general':normal_ch},
			'chb11': {'chb11_01.edf':normal_ch,
					'general':alt_ch},
			'chb13': {'general':alt_ch},
			'chb14': {'general':alt_ch},
			'chb15': {'chb15_01.edf':alt_ch,
					'general':[0,1,2,3,5,6,7,8,14,15,16,17,19,20,21,22,10,11]},
			'chb16': {'general':alt_ch},
			'chb17': {'general':alt_ch},
			'chb18': {'general':alt_ch},
			'chb19': {'general':alt_ch},
			'chb20': {'general':alt_ch},
			'chb21': {'general':alt_ch},
			'chb22': {'general':alt_ch},
			'chb23': {'general':normal_ch},
			'chb24': {'general':normal_ch}
		}
	elif dataset == 'kaggle-ieeg':
		dog_ch = [i for i in range(16)]
		CHANNELS = {
			'Patient_1': {'general':[i for i in range(15)]},
			'Patient_2': {'general':[i for i in range(24)]},
			'Dog_1': {'general':dog_ch},
			'Dog_2': {'general':dog_ch},
			'Dog_3': {'general':dog_ch},
			'Dog_4': {'general':dog_ch},
			'Dog_5': {'general':dog_ch}
		}

	return CHANNELS


def get_details(dataset):

	FS = get_fs(dataset)

	PATIENTS = get_patients(dataset)

	CHANNELS = get_channels(dataset)

	return FS, PATIENTS, CHANNELS


def get_emdmd_params(dataset):

	fs, _ , _ = get_details(dataset)
	# averaging window = 30 sec * fs samples/sec
	W = 30*fs

	# SVD truncation parameter, see estimate_r.py
	if dataset == 'chb-mit':
		R = 40
	elif dataset == 'kaggle-ieeg':
		print('\n\nERROR: R parameter not specified for this dataset. See const.get_emdmd_params().\n\n')
		exit(1)

	# cross-validation split (5 = 80 training/20 test split)
	K = 5

	# rng seed for reproducible results
	SEED = 42

	return W, R, K, SEED


def get_label_rules(FS):
	# 60 min * 60 sec/min * fs samples/sec prior to seizure
	preictal_t = 60*60*FS
	# 1 hour after seizure end
	postictal_t = 60*60*FS
	# 4 hours from any seizure
	interictal_t = 4*60*60*FS

	return preictal_t, postictal_t, interictal_t