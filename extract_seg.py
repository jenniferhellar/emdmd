import os
import argparse
import pickle
import numpy as np
import math

import util
import const


parser = argparse.ArgumentParser(
    description='Processes a single patient to extract preictal and interictal segments.')

parser.add_argument('--dataset', required=True,
					help='Dataset to process.')

parser.add_argument('--patient', required=True,
                    help='Patient to process.')

args = parser.parse_args()
dataset = vars(args)['dataset']
patient = vars(args)['patient']

if dataset not in const.DATASETS:
	print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
	exit(1)

FS, PATIENTS, CHANNELS = const.get_details(dataset)
if patient not in PATIENTS:
	print('\n\nERROR: unsupported patient argument. Select from allowed options: {} or add new options in const.get_patients()\n'.format(PATIENTS))
	exit(1)

DATADIR, RESDIR, LOGDIR = const.get_dirs(dataset)

W, _, _, _ = const.get_emdmd_params(dataset)
PRE_T, POST_T, INTER_T = const.get_label_rules(FS)

if dataset == 'chb-mit':
	fpath = os.path.join(DATADIR, patient)
elif dataset == 'kaggle-ieeg':
	fpath = os.path.join(DATADIR, patient, patient)

outDir = os.path.join(RESDIR, patient)
if not os.path.isdir(outDir):
	os.makedirs(outDir)


print('\n\nProcessing {} record {} ...'.format(dataset, patient))
files = os.listdir(fpath)

if dataset == 'chb-mit':
	ftype = '.edf'
elif dataset == 'kaggle-ieeg':
	ftype = '.mat'
	FILES_PER_HOUR	=	6

datafiles = [i for i in files if i[-4:] == ftype]
datafiles.sort()


# extra processing needed to identify preictal/interictal segments
if dataset == 'chb-mit':
	infofile = os.path.join(outDir, '{}_info.pkl'.format(patient))
	if os.path.exists(infofile):
		print('\n{} found. Loading record info ...'.format(infofile))
		with open(infofile, 'rb') as f:
			info = pickle.load(f)
		total_len = info[0]
		total_seizures = info[1]
		num_seizures = info[2]
		file_starts = info[3]
		preictal_seg = info[4]
		preictal_seg_byfiles = info[5]
		interictal_seg = info[6]
		interictal_seg_byfiles = info[7]
	else:
		print('\n{} not found. Computing record info ...'.format(infofile))
		print('\nReading ' + os.path.join(fpath, patient+'-summary.txt') + ' ...')
		summarypath = os.path.join(fpath, patient+'-summary.txt')
		infodict = util.read_summary(summarypath)

		file_starts = []
		seizure_starts = []
		seizure_stops = []

		baseidx = 0

		first = True

		for f in datafiles:
			print('\tReading ' + os.path.join(fpath, f) + ' ...')
			x_all = util.read_edf(os.path.join(fpath, f))
			if f in infodict:
				seizures = infodict[f]['s']
			else:
				seizures = 0
			s = 0
			while s < seizures:
				seizure_starts.append(infodict[f][s][0] + baseidx)
				seizure_stops.append(infodict[f][s][1] + baseidx)
				s += 1
			file_starts.append(baseidx)
			baseidx += x_all.shape[1]

		total_len = baseidx
		print('\n\n')
		print('Extracting basic file and seizure information ' + ' ...')
		print('total_len = ' + str(total_len))
		print('\n')
		print('file_starts = ', file_starts)
		print('\n')
		print('seizure_starts = ', seizure_starts)
		print('seizure_stops = ', seizure_stops)
		total_seizures = len(seizure_starts)

		""" Combine any seizures that occur <2 hour apart """
		print('\n\n')
		print('Combining seizures <2 hours apart (measured from end of 1st seizure to start of 2nd)' + ' ...')
		seizurestart = seizure_starts
		seizurestop = seizure_stops

		i = 0
		most_recent_seizure = 0
		while i < len(seizurestart) - 1:
			curr_seizure = seizurestart[i]
			if most_recent_seizure == 0:
				most_recent_seizure = seizurestop[i]
			
			next_seizure = seizurestart[i + 1]
			if (next_seizure - most_recent_seizure) < 2*60*60*256:	# 2 hours
				most_recent_seizure = seizurestop[i + 1]
				seizurestart.pop(i+1)
				seizurestop.pop(i)
			else:
				most_recent_seizure = 0
				i += 1

		print('seizurestart = ', seizurestart)
		print('seizurestop = ', seizurestop)
		num_seizures = len(seizurestart)
		print('num_seizures = ', num_seizures)	# number of seizures for this patient

		print('\n\n')
		print('Computing preictal, postictal, and interictal transitions ' + ' ...')
		preictalstart = [max(0, i - PRE_T) for i in seizurestart]
		postictalend = [i + POST_T for i in seizurestop]

		interstart = [0] + [i + INTER_T for i in seizurestop]
		interend = [max(0, i - INTER_T) for i in seizurestart] + [total_len]

		print('preictalstart = ', preictalstart)
		print('postictalend = ', postictalend)
		print('interstart = ', interstart)
		print('interend = ', interend)

		"""
		preictal segments 
		"""
		print('\n\n')
		print('Computing preictal segments ' + ' ...')

		# by global sample indices
		preictal_seg = [(preictalstart[i], seizurestart[i]) for i in range(len(seizurestart))]
		print('preictal_seg = ', preictal_seg)

		# by file breaks
		preictal_seg_byfiles = []

		for i in range(len(preictal_seg)):
			(start, stop) = preictal_seg[i]
			for f in range(len(file_starts)):
				fidx = file_starts[f]
				if start < fidx:
					if stop < fidx:
						preictal_seg_byfiles.append((start, stop))
					else:
						preictal_seg_byfiles.append((start, fidx))
						preictal_seg_byfiles.append((fidx, stop))
					break
		print('preictal_seg_byfiles = ', preictal_seg_byfiles)

		"""
		interictal segments 
		"""
		print('\n\n')
		print('Computing interictal segments ' + ' ...')

		# by global sample indices
		interictal_seg = [(interstart[i], interend[i]) for i in range(len(interstart)) if interend[i] > interstart[i]]
		print('interictal_seg = ', interictal_seg)

		# by file breaks
		interictal_seg_byfiles = []

		for i in range(len(interictal_seg)):
			(start, stop) = interictal_seg[i]

			if start < file_starts[-1]:
				startfileidx = next(x for x, val in enumerate(file_starts)
			                              if val > start)
				startfileidx -= 1
			else:
				startfileidx = len(file_starts) - 1

			stopfileidx = [i for i in range(len(file_starts)) if file_starts[i] > stop]
			if stopfileidx != []:
				stopfileidx = stopfileidx[0] - 1
			else:
				stopfileidx = len(file_starts)-1

			if stopfileidx > startfileidx:	# across multiple files
				interictal_seg_byfiles.append((start, file_starts[startfileidx+1]))
				for j in range(startfileidx+1, stopfileidx):
					interictal_seg_byfiles.append((file_starts[j], file_starts[j+1]))
				interictal_seg_byfiles.append((file_starts[stopfileidx], stop))
			else:
				interictal_seg_byfiles.append((start, stop))

		print('interictal_seg_byfiles = ', interictal_seg_byfiles)

		print('\n\nSaving all record info to ', infofile)
		with open(infofile, 'wb') as f:
			info = [total_len, total_seizures, num_seizures, file_starts,
					preictal_seg, preictal_seg_byfiles,
					interictal_seg, interictal_seg_byfiles]
			pickle.dump(info, f)


""" Extract preictal segments and save to file """
print('\n')
print('Extracting preictal segments ' + ' ...')

if dataset == 'chb-mit':
	first = True
	segidx = 0
	curr_seg = preictal_seg[segidx]
	preictal_time = 0

	for (i, j) in preictal_seg_byfiles:
		if i < file_starts[-1]:
			startfileidx = next(x for x, val in enumerate(file_starts)
		                              if val > i)
			startfileidx -= 1
		else:
			startfileidx = len(file_starts) - 1
		
		filebasesamp = file_starts[startfileidx]
		filei = i - filebasesamp	# segment start index within file
		filej = j - filebasesamp	# segment stop index within file

		f = datafiles[startfileidx]
		print('\tReading ' + os.path.join(fpath, f) + ' ...')
		x_all = util.read_edf(os.path.join(fpath, f))
		if f in CHANNELS[patient].keys():
			ch = CHANNELS[patient][f]
		else:
			ch = CHANNELS[patient]['general']
		if first:
			preictal = x_all[ch, filei:filej]	# first 18 channels
			first = False 
		else:
			# concatenate horizontally
			preictal = np.hstack((preictal, x_all[ch, filei:filej]))

		curr_seg = preictal_seg[segidx]
		if j == curr_seg[1]:
			print('\tFiltering ' + ' ...\n')
			preictal = util.butter_bandpass_filter(preictal, fs=FS)
			# truncate to a round number of minutes
			cut = math.floor(preictal.shape[1]/(FS*60))*(FS*60)
			preictal = preictal[:, -cut:]
			# save to file
			output = open(os.path.join(outDir, patient + '_preictal{}.pkl'.format(segidx)), 'wb')
			pickle.dump(preictal, output)
			output.close()

			print('\tFinished segment {} (#{} of {})'.format(curr_seg, segidx+1, len(preictal_seg)))
			print('\t\tsegment shape ', np.shape(preictal), '\n')
			preictal_time += np.shape(preictal)[1]/FS	# in seconds
			first = True
			segidx += 1

elif dataset == 'kaggle-ieeg':
	preictalfiles = [i for i in datafiles if i.find('preictal') != -1]
	num_seizures = len(preictalfiles) / FILES_PER_HOUR
	n_segs = math.ceil(len(preictalfiles)/FILES_PER_HOUR)

	cnt = 0
	segidx = 0
	preictal_time = 0
	for f in preictalfiles:
		cnt += 1

		print('\tReading ' + os.path.join(fpath, f) + ' ...')
		x_all, t_sec, fs, channels, s_idx = util.read_mat(os.path.join(fpath, f))

		if f in CHANNELS[patient].keys():
			ch = CHANNELS[patient][f]
		else:
			ch = CHANNELS[patient]['general']

		if cnt == 1:
			preictal = x_all[ch, :]
		else:
			preictal = np.hstack((preictal, x_all[ch, :]))

		if (cnt == FILES_PER_HOUR) or (f == preictalfiles[-1]):
			print('\tFiltering ...\n')
			preictal = util.butter_bandpass_filter(preictal, fs=FS)
			print('\tSaving preictal segment {} of {} ...\n'.format(segidx + 1, n_segs))
			output = open(os.path.join(outDir, patient + '_preictal{}.pkl'.format(segidx)), 'wb')
			pickle.dump(preictal, output)
			output.close()
			preictal_time += np.shape(preictal)[1]/FS	# in seconds
			segidx += 1

			cnt = 0

print('done.')


""" Extract interictal segments and save to file """
print('\n')
print('Extracting interictal segments ...')

if dataset == 'chb-mit':
	if len(interictal_seg) > 0:
		first = True
		segidx = 0
		fidx = 0
		curr_seg = interictal_seg[segidx]
		interictal_time = 0

		# limit files to 1 hour long
		max_len = 1*60*60*FS

		for (i, j) in interictal_seg_byfiles:
			if i < file_starts[-1]:
				startfileidx = next(x for x, val in enumerate(file_starts)
			                              if val > i)
				startfileidx -= 1
			else:
				startfileidx = len(file_starts) - 1

			filebasesamp = file_starts[startfileidx]
			filei = i - filebasesamp
			filej = j - filebasesamp

			f = datafiles[startfileidx]
			print('\tReading ' + os.path.join(fpath, f) + ' ...')
			x_all = util.read_edf(os.path.join(fpath, f))
			if f in CHANNELS[patient].keys():
				ch = CHANNELS[patient][f]
			else:
				ch = CHANNELS[patient]['general']
			if first:
				interictal = x_all[ch, filei:filej]
				first = False
			else:
				tmp_interictal = np.hstack((interictal, x_all[ch, filei:filej]))
				curr_len = tmp_interictal.shape[1]
				while curr_len > max_len:
					interictal = tmp_interictal[:, :max_len]
					print('\tFiltering ' + ' ...\n')
					interictal = util.butter_bandpass_filter(interictal, fs=FS)
					# save to file
					output = open(os.path.join(outDir, patient + '_interictal{}.pickle'.format(fidx)), 'wb')
					pickle.dump(interictal, output)
					output.close()

					interictal_time += interictal.shape[1]/FS	# in seconds
					fidx += 1

					print('\tFinished an hour of segment {} (#{} of {})'.format(curr_seg, segidx+1, len(interictal_seg)))
					print('\t\tsegment shape ', np.shape(interictal), '\n')

					tmp_interictal = tmp_interictal[:, max_len:]
					curr_len = tmp_interictal.shape[1]
				interictal = tmp_interictal

			# check if the end of current interictal segment is reached
			curr_seg = interictal_seg[segidx]
			if j == curr_seg[1]:
				print('\tFiltering ' + ' ...\n')
				interictal = util.butter_bandpass_filter(interictal, fs=FS)
				# truncate to a round number of minutes
				cut = math.floor(interictal.shape[1]/(FS*60))*(FS*60)
				interictal = interictal[:, -cut:]
				# save to file
				output = open(os.path.join(outDir, patient + '_interictal{}.pickle'.format(fidx)), 'wb')
				pickle.dump(interictal, output)
				output.close()

				interictal_time += interictal.shape[1]/FS	# in seconds
				fidx += 1

				print('\tFinished segment {} (#{} of {})'.format(curr_seg, segidx+1, len(interictal_seg)))
				print('\t\tsegment shape ', np.shape(interictal), '\n')

				segidx += 1
				first = True

	else:
		print('\n\n')
		print('No interictal segments to extract ' + ' ...')
		interictal_time = 0

elif dataset == 'kaggle-ieeg':
	interictalfiles = [i for i in datafiles if i.find('interictal') != -1]
	n_segs = math.ceil(len(interictalfiles)/FILES_PER_HOUR)

	cnt = 0
	segidx = 0
	interictal_time = 0
	for f in interictalfiles:
		cnt += 1

		print('\t\tReading ' + os.path.join(fpath, f) + ' ...')
		x_all, t_sec, fs, channels, s_idx = util.read_mat(os.path.join(fpath, f))
		if f in CHANNELS[patient].keys():
			ch = CHANNELS[patient][f]
		else:
			ch = CHANNELS[patient]['general']

		if cnt == 1:
			interictal = x_all[ch, :]
		else:
			interictal = np.hstack((interictal, x_all[ch, :]))

		if (cnt == FILES_PER_HOUR) or (f == interictalfiles[-1]):
			print('\tFiltering ...\n')
			interictal = util.butter_bandpass_filter(interictal, fs=FS)
			print('\tSaving interictal segment {} of {} ...\n'.format(segidx + 1, n_segs))
			output = open(os.path.join(outDir, patient + '_interictal{}.pickle'.format(segidx)), 'wb')
			pickle.dump(interictal, output)
			output.close()
			interictal_time += np.shape(interictal)[1]/FS	# in seconds
			segidx += 1

			cnt = 0

print('done.')


print('\n\nOutput files can be found in {}\n'.format(outDir))
if dataset == 'chb-mit':
	print('total_seizures = ', total_seizures)
print('num_seizures = ', num_seizures)	# number of seizures for this patient
print('preictal time (min) = ', preictal_time/60)	# in min
print('interictal time (min) = ', interictal_time/60)	# in min