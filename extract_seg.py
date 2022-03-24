"""
A script to label, extract, and save preictal/interictal segments
from the requested EEG or iEEG patient.
-----------------------------------------------------------------------------

file: extract_seg.py
author: Jennifer Hellar
email: jenniferhellar@pm.me

packages: os, argparse, pickle, numpy, math
local dependencies: util.py and const.py

-----------------------------------------------------------------------------
usage: extract_seg.py [-h] --dataset DATASET --patient PATIENT

Extracts, downsamples, and filters preictal and interictal segments for the input patient.

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name e.g. chb-mit, kaggle-ieeg
  --patient PATIENT  Patient name e.g. chb01, Patient_1

-----------------------------------------------------------------------------
Detailed description
-----------------------------------------------------------------------------
Labelling:
- CHB-MIT: Identify preictal and interictal segments based on seizure 
info and labelling rules specified in const.get_label_rules()
- Kaggle-IEEG: Specified by file names (pre-labelled)

Extraction and processing:
- Split into at most 1-hour long segments
- For each segment,
	- Downsample by the factor specified for the dataset in const.py
	- Filter with a 2nd-order Butterworth bandpass filter from 0.1 to 40 Hz 

Creates output directory 'tempdata/[dataset]/[patient]/'
	- Saves files by naming convention [patient]_[state][index].pkl 
		e.g. chb01_preictal0.pkl
	- CHB-MIT: Additionally saves [patient]_info.pkl with preictal/interictal
		segment indices and file start/stop indices to skip above labelling
		step after first run.
-----------------------------------------------------------------------------
"""
import os
import argparse
import pickle
import numpy as np
import math

# local scripts
import util
import const


parser = argparse.ArgumentParser(
    description='Extracts, downsamples, and filters preictal and interictal segments for the input patient.')

parser.add_argument('--dataset', required=True,
					help='Dataset name e.g. chb-mit, kaggle-ieeg')

parser.add_argument('--patient', required=True,
                    help='Patient name e.g. chb01, Patient_1')

args = parser.parse_args()
dataset = vars(args)['dataset']
patient = vars(args)['patient']

if dataset not in const.DATASETS:
	print('\n\nERROR: unsupported dataset argument. Allowed options: \n', const.DATASETS)
	exit(1)

PATIENTS = const.PATIENTS[dataset]
if patient not in PATIENTS:
	print('\n\nERROR: unsupported patient argument. Select from allowed options: {} or add new options in const.py\n'.format(PATIENTS))
	exit(1)

# dataset original sampling frequency
ORIG_FS = const.ORIG_FS[dataset]
# channels to include for this patient
CHANNELS = const.CHANNELS[dataset][patient]
# data directory and output directory
DATADIR, RESDIR, _ = const.get_dirs(dataset)

# downsampling factor and effective sampling frequency after downsampling
DS_FACTOR = const.DS_FACTOR[dataset]
FS = const.get_fs(dataset)

# preictal samples prior to seizure
# postictal samples after seizure
# interictal exclusion samples before and after seizure
PRE_T, POST_T, INTER_T = const.get_label_rules(ORIG_FS)

if dataset == 'chb-mit':
	fpath = os.path.join(DATADIR, patient)
elif dataset == 'kaggle-ieeg':
	fpath = os.path.join(DATADIR, patient, patient)

OUTDIR = os.path.join(RESDIR, patient)
if not os.path.isdir(OUTDIR):
	os.makedirs(OUTDIR)


print('\n\nProcessing {} record {} ...'.format(dataset, patient))
files = os.listdir(fpath)

if dataset == 'chb-mit':
	ftype = '.edf'
elif dataset == 'kaggle-ieeg':
	ftype = '.mat'
	# files saved in 10-min segments
	FILES_PER_HOUR	=	6

datafiles = [i for i in files if i[-4:] == ftype]
datafiles.sort()


# CHB-MIT: extra processing needed to identify preictal/interictal segments
if dataset == 'chb-mit':
	infofile = os.path.join(OUTDIR, '{}_info.pkl'.format(patient))
	if os.path.exists(infofile):
		print('Record info found. Loading record info from\n\t{} ...'.format(infofile))
		with open(infofile, 'rb') as f:
			info = pickle.load(f)
		total_len = info[0]			# total number of samples for this patient
		total_seizures = info[1]	# total number of seizures
		num_seizures = info[2]		# seizures usable (excluding any <2 hours apart)
		file_starts = info[3]		# sample nums of new file starts in terms of global idx
		preictal_seg = info[4]		# list of tuples (start, stop) of preictal segments
		preictal_seg_byfiles = info[5]		# list of preictal segment tuples split by file breaks
		interictal_seg = info[6]	# list of tuples (start, stop) of interictal segments
		interictal_seg_byfiles = info[7]	# list of interictal segment tuples split by file breaks
	else:
		print('\nRecord info not found at\n\t{}\nComputing record info ...'.format(infofile))
		print('\nReading ' + os.path.join(fpath, patient+'-summary.txt') + ' ...')
		summarypath = os.path.join(fpath, patient+'-summary.txt')
		# info dictionary structure
		#	{filename: {'s': numseizures, 
		#				seizurenumber: (seizure_start, seizure_stop)}}
		infodict = util.read_summary(summarypath)

		file_starts = []
		seizure_starts = []
		seizure_stops = []

		# to track segments across file breaks, use global
		# sample indices that increment over sequential files
		#	e.g. if last sample of record x is 1000, then first sample
		#		of record x+1 is 1001

		# global index of sample start for current file
		baseidx = 0

		first = True

		for f in datafiles:
			print('\tReading ' + os.path.join(fpath, f) + ' ...')
			x_all = util.read_edf(os.path.join(fpath, f))
			# check the number of seizures in this file
			if f in infodict:
				seizures = infodict[f]['s']
			else:
				seizures = 0
			# for each seizure
			s = 0
			while s < seizures:
				# add start/stop index within file to global baseidx of the file
				seizure_starts.append(infodict[f][s][0] + baseidx)
				seizure_stops.append(infodict[f][s][1] + baseidx)
				s += 1
			file_starts.append(baseidx)
			# increment baseidx to start of next file
			baseidx += x_all.shape[1]

		total_len = baseidx	# now this is the total num of samples for patient
		print('\n\n')
		print('Extracting basic file and seizure information ' + ' ...')
		print('total_len = ' + str(total_len))
		print('\n')
		# global index of each file start
		print('file_starts = ', file_starts)	
		print('\n')
		# lists of global indices of seizure start/stops
		print('seizure_starts = ', seizure_starts)
		print('seizure_stops = ', seizure_stops)
		# total number of seizure for patient
		total_seizures = len(seizure_starts)

		""" Combine any seizures that occur <2 hour apart """
		print('\n\n')
		print('Combining seizures <2 hours apart (measured from end of 1st seizure to start of 2nd)' + ' ...')
		seizurestart = seizure_starts
		seizurestop = seizure_stops

		i = 0
		most_recent_seizure = 0
		while i < len(seizurestart) - 1:
			# end of current seizure
			if most_recent_seizure == 0:
				most_recent_seizure = seizurestop[i]
			# next seizure start
			next_seizure = seizurestart[i + 1]

			# if <2 hours apart
			if (next_seizure - most_recent_seizure) < 2*60*60*ORIG_FS:	# 2 hours
				# update end to the end of the combined seizures
				most_recent_seizure = seizurestop[i + 1]
				# combine by removing end of current and start of next
				seizurestart.pop(i+1)
				seizurestop.pop(i)
			else:
				most_recent_seizure = 0
				i += 1

		print('seizurestart = ', seizurestart)
		print('seizurestop = ', seizurestop)
		# number of usable seizures
		num_seizures = len(seizurestart)
		print('num_seizures = ', num_seizures)	# number of seizures for this patient

		print('\n\n')
		print('Computing preictal, postictal, and interictal transitions ' + ' ...')
		# preictal period starts PRE_T samples before seizure onset
		preictalstart = [max(0, i - PRE_T) for i in seizurestart]
		# postictal period ends POST_T samples after seizure end
		postictalend = [i + POST_T for i in seizurestop]

		# interictal periods start INTER_T samples after seizure end
		interstart = [0] + [i + INTER_T for i in seizurestop]
		# they end INTER_T samples prior to seizure onset (or at the record start)
		#	or at the end of the patient records
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

		# tuples of (start, stop) by global sample indices
		preictal_seg = [(preictalstart[i], seizurestart[i]) for i in range(len(seizurestart))]
		print('preictal_seg = ', preictal_seg)

		# split up tuples by file breaks
		preictal_seg_byfiles = []

		for i in range(len(preictal_seg)):
			(start, stop) = preictal_seg[i]
			for f in range(len(file_starts)):
				# sample index of file break
				fidx = file_starts[f]
				if start < fidx:
					if stop < fidx:
						# entire segment before file break
						preictal_seg_byfiles.append((start, stop))
					else:
						# segment starts prior to break and ends after
						preictal_seg_byfiles.append((start, fidx))
						preictal_seg_byfiles.append((fidx, stop))
					break
		print('preictal_seg_byfiles = ', preictal_seg_byfiles)

		"""
		interictal segments 
		"""
		print('\n\n')
		print('Computing interictal segments ' + ' ...')

		# tuples of (start, stop) segments by global sample indices
		interictal_seg = [(interstart[i], interend[i]) for i in range(len(interstart)) if interend[i] > interstart[i]]
		print('interictal_seg = ', interictal_seg)

		# by file breaks
		interictal_seg_byfiles = []

		for i in range(len(interictal_seg)):
			(start, stop) = interictal_seg[i]

			if start < file_starts[-1]:
				# starts in some file before the last record,
				# so find the file that the segment starts in
				startfileidx = next(x for x, val in enumerate(file_starts)
			                              if val > start)
				startfileidx -= 1
			else:
				# starts in the last file
				startfileidx = len(file_starts) - 1

			# find the file that the segment ends in
			stopfileidx = [i for i in range(len(file_starts)) if file_starts[i] > stop]
			if stopfileidx != []:
				stopfileidx = stopfileidx[0] - 1
			else:
				# stops in the last file
				stopfileidx = len(file_starts)-1

			if stopfileidx > startfileidx:	# across multiple files
				# start of segment to end of first file
				interictal_seg_byfiles.append((start, file_starts[startfileidx+1]))
				# entire files
				for j in range(startfileidx+1, stopfileidx):
					interictal_seg_byfiles.append((file_starts[j], file_starts[j+1]))
				# start of last file to end of segment
				interictal_seg_byfiles.append((file_starts[stopfileidx], stop))
			else:
				interictal_seg_byfiles.append((start, stop))

		print('interictal_seg_byfiles = ', interictal_seg_byfiles)

		# save the key information to a file
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

	# each tuple of (start, stop)
	for (i, j) in preictal_seg_byfiles:
		# find the file that contains this segment
		if i < file_starts[-1]:
			startfileidx = next(x for x, val in enumerate(file_starts)
		                              if val > i)
			startfileidx -= 1
		else:
			startfileidx = len(file_starts) - 1
		
		filebasesamp = file_starts[startfileidx]
		filei = i - filebasesamp	# segment start index within file
		filej = j - filebasesamp	# segment stop index within file

		# read in the data file
		f = datafiles[startfileidx]
		print('\tReading ' + os.path.join(fpath, f) + ' ...')
		x_all = util.read_edf(os.path.join(fpath, f))

		# get the channels to use for this patient
		#	(sometimes channels change across files)
		if f in CHANNELS.keys():
			ch = CHANNELS[f]
		else:
			ch = CHANNELS['general']

		if first:
			# first part of the preictal segment
			preictal = x_all[ch, filei:filej]
			first = False 
		else:
			# otherwise, concatenate horizontally (across time)
			preictal = np.hstack((preictal, x_all[ch, filei:filej]))

		# if we've found the end of this preictal segment ...
		curr_seg = preictal_seg[segidx]
		if j == curr_seg[1]:
			print('\tSub-sampling ...\n')
			preictal = util.downsample(preictal, DS_FACTOR)

			print('\tFiltering ' + ' ...\n')
			preictal = util.butter_bandpass_filter(preictal, fs=FS)
			# truncate to a round number of minutes
			cut = math.floor(preictal.shape[1]/(FS*60))*(FS*60)
			preictal = preictal[:, -cut:]
			# save to file
			output = open(os.path.join(OUTDIR, patient + '_preictal{}.pkl'.format(segidx)), 'wb')
			pickle.dump(preictal, output)
			output.close()

			print('\tFinished segment {} (#{} of {})'.format(curr_seg, segidx+1, len(preictal_seg)))
			print('\t\tsegment shape ', np.shape(preictal), '\n')
			preictal_time += np.shape(preictal)[1]/FS	# in seconds
			first = True
			segidx += 1

elif dataset == 'kaggle-ieeg':
	preictalfiles = [i for i in datafiles if i.find('preictal') != -1]
	# one hour of preictal data per seizure
	num_seizures = len(preictalfiles) / FILES_PER_HOUR
	n_segs = math.ceil(len(preictalfiles)/FILES_PER_HOUR)

	cnt = 0
	segidx = 0
	preictal_time = 0
	for f in preictalfiles:
		cnt += 1

		print('\tReading ' + os.path.join(fpath, f) + ' ...')
		x_all, _, _, _, _ = util.read_mat(os.path.join(fpath, f))

		# get the channels to use for this patient
		if f in CHANNELS.keys():
			ch = CHANNELS[f]
		else:
			ch = CHANNELS['general']

		if cnt == 1:
			# first part of this preictal segment
			preictal = x_all[ch, :]
		else:
			# stack horizontally (across time)
			preictal = np.hstack((preictal, x_all[ch, :]))

		# if end of this segment or last of the files ...
		if (cnt == FILES_PER_HOUR) or (f == preictalfiles[-1]):
			print('\tSub-sampling ...\n')
			preictal = util.downsample(preictal, DS_FACTOR)
			
			print('\tFiltering ...\n')
			preictal = util.butter_bandpass_filter(preictal, fs=FS)

			print('\tSaving preictal segment {} of {} ...\n'.format(segidx + 1, n_segs))
			output = open(os.path.join(OUTDIR, patient + '_preictal{}.pkl'.format(segidx)), 'wb')
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
		max_len = 1*60*60*ORIG_FS

		for (i, j) in interictal_seg_byfiles:
			# find file where current segment starts
			if i < file_starts[-1]:
				startfileidx = next(x for x, val in enumerate(file_starts)
			                              if val > i)
				startfileidx -= 1
			else:
				startfileidx = len(file_starts) - 1

			filebasesamp = file_starts[startfileidx]
			# segment start/stop sample indices within file
			filei = i - filebasesamp
			filej = j - filebasesamp

			# open and read the data file
			f = datafiles[startfileidx]
			print('\tReading ' + os.path.join(fpath, f) + ' ...')
			x_all = util.read_edf(os.path.join(fpath, f))
			# get the channels to use
			if f in CHANNELS.keys():
				ch = CHANNELS[f]
			else:
				ch = CHANNELS['general']
			if first:
				# first part of the current segment
				interictal = x_all[ch, filei:filej]
				first = False
			else:
				# concatenate across time
				tmp_interictal = np.hstack((interictal, x_all[ch, filei:filej]))
				# if length > 1 hour, save in 1-hour sub-segments
				curr_len = tmp_interictal.shape[1]
				while curr_len > max_len:
					# grab the next hour
					interictal = tmp_interictal[:, :max_len]
					print('\tSub-sampling ...\n')
					interictal = util.downsample(interictal, DS_FACTOR)
					print('\tFiltering ' + ' ...\n')
					interictal = util.butter_bandpass_filter(interictal, fs=FS)
					# save to file
					output = open(os.path.join(OUTDIR, patient + '_interictal{}.pkl'.format(fidx)), 'wb')
					pickle.dump(interictal, output)
					output.close()

					interictal_time += interictal.shape[1]/FS	# in seconds
					fidx += 1

					print('\tFinished an hour of segment {} (#{} of {})'.format(curr_seg, segidx+1, len(interictal_seg)))
					print('\t\tsegment shape ', np.shape(interictal), '\n')

					# remove the hour just saved
					tmp_interictal = tmp_interictal[:, max_len:]
					curr_len = tmp_interictal.shape[1]
				# if < 1 hour, just continue
				interictal = tmp_interictal

			# check if the end of current interictal segment is reached
			curr_seg = interictal_seg[segidx]
			if j == curr_seg[1]:
				print('\tSub-sampling ...\n')
				interictal = util.downsample(interictal, DS_FACTOR)
				print('\tFiltering ' + ' ...\n')
				interictal = util.butter_bandpass_filter(interictal, fs=FS)
				# truncate to a round number of minutes
				cut = math.floor(interictal.shape[1]/(FS*60))*(FS*60)
				interictal = interictal[:, -cut:]
				# save to file
				output = open(os.path.join(OUTDIR, patient + '_interictal{}.pkl'.format(fidx)), 'wb')
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
		# read in the data file
		print('\t\tReading ' + os.path.join(fpath, f) + ' ...')
		x_all, t_sec, _, _, s_idx = util.read_mat(os.path.join(fpath, f))
		# get the correct channels to use
		if f in CHANNELS.keys():
			ch = CHANNELS[f]
		else:
			ch = CHANNELS['general']

		if cnt == 1:
			# first part of the segment
			interictal = x_all[ch, :]
		else:
			# concatenate across time dimension
			interictal = np.hstack((interictal, x_all[ch, :]))

		# reached end of the segment or the final file
		if (cnt == FILES_PER_HOUR) or (f == interictalfiles[-1]):
			print('\tSub-sampling ...\n')
			interictal = util.downsample(interictal, DS_FACTOR)
			print('\tFiltering ...\n')
			interictal = util.butter_bandpass_filter(interictal, fs=FS)
			print('\tSaving interictal segment {} of {} ...\n'.format(segidx + 1, n_segs))
			output = open(os.path.join(OUTDIR, patient + '_interictal{}.pkl'.format(segidx)), 'wb')
			pickle.dump(interictal, output)
			output.close()
			interictal_time += np.shape(interictal)[1]/FS	# in seconds
			segidx += 1

			cnt = 0

print('done.')


print('\n\nOutput files can be found in {}\n'.format(OUTDIR))
if dataset == 'chb-mit':
	print('total_seizures = ', total_seizures)
print('num_seizures = ', num_seizures)	# number of seizures for this patient
print('preictal time (min) = ', preictal_time/60)	# in min
print('interictal time (min) = ', interictal_time/60)	# in min