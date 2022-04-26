# Epileptic EEG classification using Embedded Dynamic Mode Decomposition
Author: Jennifer Hellar \
Date: Apr 26, 2022

## Datasets
CHB-MIT scalp EEG \
    (https://physionet.org/content/chbmit/1.0.0/) \
Kaggle American Epilepsy Society Seizure Prediction Challenge intracranial EEG \
    (https://www.kaggle.com/c/seizure-prediction)

## Key scripts:
const.py \
    Dataset constants, key EmDMD and simulation parameters, and functions to 
    obtain directory paths, effective sampling frequencies, and
    preictal/interictal labelling rules.

emdmd.py \
    Embedded Dynamic Mode Decomposition (EmDMD) library.

util.py \
    Common utility library.

## Process entire datasets:
extract_all_seg.py [-h] --dataset DATASET \
    Extracts preictal/interictal segments from all patients in the input dataset.

extract_all_feat.py [-h] --dataset DATASET \
    Processes all patients from the input dataset to compute EmDMD features.

classify_patient_specific.py [-h] --dataset DATASET --sph SPH --sop SOP [-v VERBOSE] \
    Classifies each patient in the input dataset and prints CV results.

## Process individual patients:
extract_seg.py [-h] --dataset DATASET --patient PATIENT \
    Extracts and downsamples preictal and interictal segments for the input patient.

extract_feat.py [-h] --dataset DATASET --patient PATIENT [--class CLASS] [--index INDEX] \
    Computes EmDMD features from the preictal/interictal segment(s) of the input patient.

split_kfolds.py [-h] --dataset DATASET [--patient PATIENT] --sph SPH --sop SOP \
    Creates training/test cross-validation splits on the computed EmDMD features of the input dataset.

classify.py [-h] --dataset DATASET [-f FOLD] \
    Classifies the test data of the input dataset to measure performance.

## Other scripts:
freq_tb.py \
    Computes the EmDMD frequencies present in subsets of the data (to detemine r parameter needed).

make_figures.py \
    Creates the figures present in the paper and provides other misc simulations.
