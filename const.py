import os
import platform


mySys = platform.system()

if mySys == 'Linux':
	usr = os.environ.get('USER')
	CODEDIR = '/home/' + usr + '/emdmd'
	if usr == 'jlh24':
		WORKDIR = '/media/large_disk/jlh24/emdmd'
	else:
		WORKDIR = CODEDIR
	DATADIR = os.path.join(WORKDIR, 'data', 'physionet.org/files/chbmit/1.0.0')
elif mySys == 'Windows':
	CODEDIR =  'D:\\EmDMD\\emdmd'
	WORKDIR = 'D:\\EmDMD\\emdmd'
	DATADIR = os.path.join(WORKDIR, 'data', 'chb-mit')
else:
	print('\nERROR: unknown operating system. Must specify workDir and dataDir in const.py\n')
	exit(1)

RESDIR = os.path.join(CODEDIR, 'tempdata', 'chb-mit')
LOGDIR = os.path.join(CODEDIR, 'logs')

FS = 256

PRE_T = 60*60*FS		# 60 min * 60 sec/min * fs samples/sec prior to seizure
POST_T = 60*60*FS		# 1 hour after seizure end
INTER_T = 4*60*60*FS	# 4 hours from any seizure
W = 30*FS 				# window = 30 sec * fs samples/sec

R = 40					# SVD truncation parameter, see tb_estimate_r.py

SEED = 42

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

PATIENTS = ['chb01', 'chb02', 'chb04', 'chb05', 'chb06', 'chb07', 'chb09',
			'chb10', 'chb13', 'chb14', 'chb16', 'chb17', 'chb18', 'chb20',
			'chb22', 'chb23']

# PATIENTS = ['chb01']
