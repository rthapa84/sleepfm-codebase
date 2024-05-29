import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data

from tqdm import tqdm
from utils import *
import multiprocessing
from typing import List, Tuple
import csv
from loguru import logger
import warnings
import pickle

import sys
import config
from config import ALL_CHANNELS, PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA
from scipy.io import loadmat
import h5py
import glob
import scipy.signal
import os
import argparse

EVENT_TO_ID = {
    "wake": 1, 
    "nonrem1": 2, 
    "nonrem2": 3, 
    "nonrem3": 4, 
    "rem": 5, 
}
ID_TO_EVENT = {value: key for key, value in EVENT_TO_ID.items()}


# Ignore all warnings
warnings.filterwarnings("ignore")

def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

def extract_labels(path):
    data = h5py.File(path, 'r')
    length = data['data']['sleep_stages']['wake'].shape[1]
    labels = np.zeros((length, 6)) 

    for i, label in enumerate(data['data']['sleep_stages'].keys()):
        labels[:,i] = data['data']['sleep_stages'][label][:]
    
    return labels, list(data['data']['sleep_stages'].keys())

def resample_signal(data, labels, old_fs):
    diff = np.diff(labels, axis = 0)
    cutoff = np.where(diff[:,4] != 0)[0]+1
    data, labels = data[cutoff[0]+1:,:], labels[cutoff[0]+1:,:]

    new_fs = 100
    num = int(len(data)/(old_fs/new_fs))
    resampled_data = scipy.signal.resample(data, num = num, axis = 0)
    resampled_labels = labels[::int((old_fs/new_fs)),:]
    return resampled_data.astype(np.int16), resampled_labels.astype(np.int16)


def preprocess_EEG(folder,remove_files = False, out_folder = None):
    files = glob.glob(f'{folder}/*')
    data = None
    labels = None
    Fs = None
    for file in files:
        if '.hea' in file:
            s, Fs, n_samples = import_signal_names(file)
            if remove_files:
                os.remove(file)
        elif '-arousal.mat' in file:
            labels, label_names = extract_labels(file)
            if remove_files:
                os.remove(file)
        elif 'mat' in file:
            data = loadmat(file)['val'][:6, :]
            if remove_files:
                os.remove(file)

    if not data is None:
        diff = np.diff(labels, axis = 0)
        cutoff = np.where(diff[:,4] != 0)[0]+1
        data, labels = data[:, cutoff[0]+1:], labels[cutoff[0]+1:,:]

        info = mne.create_info(s[:6], Fs, ch_types = 'eeg')
        mne_dataset = mne.io.RawArray(data, info)

        events = process_labels_to_events(labels, label_names)
        label_dict = dict(zip(np.arange(0,6), label_names))
        events = np.array(events)
        event_dict = dict(zip(label_names, np.arange(0,6)))
        f = lambda x: label_dict[x]
        annotations = mne.Annotations(onset = events[:,0]/Fs, duration = events[:,1]/Fs, description  = list(map(f,events[:,2])))
        mne_dataset.set_annotations(annotations)

        mne_dataset.resample(sfreq = 100)
        epoch_events = mne.events_from_annotations(mne_dataset, chunk_duration = 30)
        info = mne.create_info(['STI'], mne_dataset.info['sfreq'], ['stim'])
        stim_data = np.zeros((1, len(mne_dataset.times)))
        stim_raw = mne.io.RawArray(stim_data, info)
        mne_dataset.add_channels([stim_raw], force_update_info=True)
        mne_dataset.add_events(epoch_events[0], stim_channel = 'STI')
        mne_dataset.save(f'{out_folder}/001_30s_raw.fif', overwrite = True)

def relocate_EEG_data(folder, remove_files = True):

    data_file = mne.read_epochs(f'{folder}/001_30s.fif')
    #h5py.File(f'{folder}/data.hdf5', 'r')
    new_name = f'{folder}/001_30s_epo.fif'
    data_file.save(new_name)
    if remove_files:
        os.remove(f'{folder}/data.mat')
        os.remove(f'{folder}/001_30s.fif')

def process_labels_to_events(labels, label_names):
    new_labels = np.argmax(labels, axis = 1)
    lab = new_labels[0]
    events = []
    start = 0
    i = 0
    while i < len(new_labels)-1:
        while new_labels[i] == lab and i < len(new_labels)-1:
            i+=1
        end = i
        dur = end +1 - start
        events.append([start, dur, lab])
        lab = new_labels[i]
        start = i+1
    return events


def get_arguments():
    parser = argparse.ArgumentParser(description="Process data and save to files")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to the EDF files")
    parser.add_argument("--save_path", type=str, default=None, 
                        help="Path to save preprocessed data")
    parser.add_argument("--num_files", type=int, default=10, 
                        help="Number of files to process")
    parser.add_argument("--chunk_duration", type=float, default=30.0,
                        help="Duration of data chunks in seconds")
    parser.add_argument("--num_threads", type=int, default=4,
                            help="Number of threads for parallel processing")
    parser.add_argument("--target_sampling_rate", type=int, default=256,
                            help="Target Sampling of the dataset")
          
    return parser.parse_args()

def parallel_process_edf_file(args: Tuple[List[str], str, float]):

    edf_and_event_files = args[0]
    path_to_save = args[1]
    chunk_duration = args[2]
    target_sampling_rate = args[3]

    path_to_X = os.path.join(path_to_save, "X")
    path_to_Y = os.path.join(path_to_save, "Y")

    for edf_dict in tqdm(edf_and_event_files):

        arousal_mat_filename = edf_dict["arousal_mat"]
        hea_filename = edf_dict["hea"]
        edf_filename = edf_dict["mat"]

        file_prefix = edf_filename.split("/")[-1].split(".")[0]

        path_to_patient_X = os.path.join(path_to_X, file_prefix)
        path_to_patient_Y = os.path.join(path_to_Y, f"{file_prefix}.pickle")

        if os.path.exists(path_to_patient_X) and len(os.listdir(path_to_patient_X)) > 0:
            logger.info(f"Patient already processed: {path_to_patient_X}")
            continue

        try:
            s, Fs, n_samples = import_signal_names(hea_filename)
            labels, label_names = extract_labels(arousal_mat_filename)
            data = loadmat(edf_filename)['val']
        except:
            continue

        if data is None:
            continue

        try:
            diff = np.diff(labels, axis = 0)
            cutoff = np.where(diff[:,4] != 0)[0]+1
            data, labels = data[:, cutoff[0]+1:], labels[cutoff[0]+1:,:]
            info = mne.create_info(s, Fs, ch_types = 'eeg')
            edf_raw = mne.io.RawArray(data, info)

            events = process_labels_to_events(labels, label_names)
            label_dict = dict(zip(np.arange(0,6), label_names))
            events = np.array(events)
            event_dict = dict(zip(label_names, np.arange(0,6)))
            f = lambda x: label_dict[x]
            annotations = mne.Annotations(onset = events[:,0]/Fs, duration = events[:,1]/Fs, description  = list(map(f,events[:,2])))
            edf_raw.set_annotations(annotations, emit_warning=True)
            edf_raw.resample(sfreq = 256)

            sfreq = edf_raw.info["sfreq"]
            logger.info(f"Original sfreq: {sfreq}")

            events_raw, _ = mne.events_from_annotations(edf_raw, event_id=EVENT_TO_ID, chunk_duration=chunk_duration)
            event_ids = set(events_raw[:, 2])
            true_event_to_id = {ID_TO_EVENT[idx]: idx for idx in event_ids}
            tmax = chunk_duration - 1.0 / sfreq  # tmax in included
            epochs = mne.Epochs(
                raw=edf_raw,
                events=events_raw,
                event_id=true_event_to_id,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
                event_repeated='drop',
            ).drop_bad()
        except Exception as e:
            print(f"Warning: An error occurred - {e}")
            continue
        
        indices = [edf_raw.ch_names.index(channel) for channel in ALL_CHANNELS if channel in edf_raw.ch_names]
    
        labels = {}

        path_to_patient_X = os.path.join(path_to_X, file_prefix)
        path_to_patient_Y = os.path.join(path_to_Y, f"{file_prefix}.pickle")

        os.makedirs(path_to_patient_X, exist_ok=True)

        for idx, epoch in enumerate(epochs):
            data = epoch[indices, :]
            assert data.shape[0] == len(indices)

            if data.shape[0] != len(ALL_CHANNELS) or data.shape[1] == 0:
                continue

            num_samples_target = int(data.shape[1] * target_sampling_rate / sfreq)
            resampled_data = scipy.signal.resample(data, num_samples_target, axis=1)
            
            file_name = f"{file_prefix}_{idx}.npy"
            np.save(os.path.join(path_to_patient_X, file_name), resampled_data)

            labels[file_name] = epochs[idx].event_id

        with open(path_to_patient_Y, 'wb') as file:
            pickle.dump(labels, file)

    return [1]


def main():
    args = get_arguments()

    path_to_edf_files = args.data_path
    path_to_save = args.save_path

    # you can also define the file paths in config
    if path_to_edf_files == None:
        path_to_edf_files = PATH_TO_RAW_DATA
    if path_to_save == None:
        path_to_save = PATH_TO_PROCESSED_DATA

    num_files = args.num_files
    chunk_duration = args.chunk_duration
    num_threads = args.num_threads
    target_sampling_rate = args.target_sampling_rate

    os.makedirs(path_to_save, exist_ok=True)

    path_to_X = os.path.join(path_to_save, "X")
    path_to_Y = os.path.join(path_to_save, "Y")

    os.makedirs(path_to_X, exist_ok=True)
    os.makedirs(path_to_Y, exist_ok=True)

    data_dict = {}

    subjects = os.listdir(path_to_edf_files)
    subjects_path = [os.path.join(path_to_edf_files, subject) for subject in subjects]

    edf_events_files_pruned = []
    for subject_path in subjects_path:
        if "RECORDS" in subject_path or "ANNOTATORS" in subject_path or len(os.listdir(subject_path)) != 4:
            continue

        files = glob.glob(f'{subject_path}/*')
        try:
            temp_dict = {}
            for file in files:
                if '.hea' in file:
                    s, Fs, n_samples = import_signal_names(file)
                    temp_dict["hea"] = os.path.join(file)
                elif '-arousal.mat' in file:
                    temp_dict["arousal_mat"] = os.path.join(file)
                elif 'mat' in file:
                    temp_dict["mat"] = os.path.join(file)
            edf_events_files_pruned.append(temp_dict)
        except:
            continue

    logger.info(f"Starting Extraction...")

    if num_files != -1:
        edf_events_files_pruned = edf_events_files_pruned[:num_files]

    edf_and_event_files_per_thread = np.array_split(edf_events_files_pruned, num_threads)

    tasks = [(edf_and_event_file, path_to_save, chunk_duration, target_sampling_rate) for edf_and_event_file in edf_and_event_files_per_thread]
    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_results = [
            y for x in pool.imap_unordered(parallel_process_edf_file, tasks) for y in x
        ]

if __name__ == "__main__":
    main()