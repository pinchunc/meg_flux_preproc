import os.path as op
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import scipy.io

import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids
from autoreject import get_rejection_threshold
import numpy as np
# import settings
from FLUXSettings import bids_root,task_list,session
get_ipython().run_line_magic("matplotlib", "qt")
bids_root = '/Volumes/MEG_BIDS/BIDS-data/'

resample_sfreq = 500;
l_freq = 0.35
h_freq = 200 # for gamma
iir_params = dict(order=5, ftype='butter')
notch_filter_freqs = (50, 100, 150) # must be lowe than lowpass
task = 'sleep'
meg_suffix = 'meg'
ica_suffix = 'ica'
epo_suffix = 'epo'
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

sub_list = ['106', '107', '112', '118', '110', '111', '116', '117', '119']#'102', '103', '104', 
# 
def is_overlap(annot1, annot2): # add a 250ms buffer before and after
    annot1_end = annot1['onset'] + annot1['duration'] + 0.25
    annot2_end = annot2['onset'] + annot2['duration'] + 0.25
    return not (annot1['onset'] - 0.25 >= annot2_end or annot2['onset'] - 0.25 >= annot1_end)

for subject in sub_list:
    bids_path = BIDSPath(subject=subject, session=session,
                task=task, run=1, suffix=epo_suffix, datatype='meg',
                root=deriv_root, extension='.fif', check=False).mkdir()
    
    deriv_file_nrem = bids_path.basename.replace('run-01', 'run-all-nrem-5sec')  # run 12 -> run 01 concatenated with run 02
    deriv_fname_nrem = op.join(bids_path.directory, deriv_file_nrem)
    
    # loop through runs
    
    raw_list = []
    events_list = []
    
    task = 'sleep'
    files = os.listdir(os.path.join(preproc_root, f'sub-{subject}', f'ses-{session}', 'meg'))
    task_files = [file for file in files if file.endswith('ica.fif') and file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]
    for run_idx in range(len(task_files)):
        run = run_idx+1
        bids_path_preproc = BIDSPath(subject=subject, session=session,
                                     task=task, run=run, suffix=meg_suffix, datatype='meg',
                                     root=preproc_root, extension='.fif', check=False)
    
        bids_path = BIDSPath(subject=subject, session=session,
                             task=task, run=run, suffix=epo_suffix, datatype='meg',
                             root=deriv_root, extension='.fif', check=False).mkdir()
    
        print(bids_path_preproc.fpath)
        
        # Create event markers
        ica_path = BIDSPath(subject=subject, session=session,
                            task=task, run=run, suffix=ica_suffix, datatype='meg',
                            root=preproc_root, extension='.fif', check=False).mkdir()
        raw = mne.io.read_raw_fif(ica_path, allow_maxshield=True, verbose=True, preload=False)
        sfreq = raw.info['sfreq']  # should be 1000 Hz in your case    
        decim = np.round(sfreq / resample_sfreq).astype(int)
        obtained_sfreq = sfreq / decim
        lowpass_freq = obtained_sfreq / 3.0
        
        stage_fname = f'{bids_root}/sleep_edf/s{subject}/auto_stage_s{subject}_r{run}.mat'
        stage_data = scipy.io.loadmat(stage_fname)
        stage_data = stage_data['stageData']['stages'][0][0]
        
        
        # Create event markers
        epoch_duration = 30  # seconds
        stages_events = []
        for i, stage in enumerate(stage_data):
            start_sample = int(i * epoch_duration * sfreq) + raw.first_samp
            stages_events.append([start_sample, 0, stage[0]])
        
        stages_events = np.array(stages_events)
        event_indices = np.where((stages_events[:, 2] == 2) | (stages_events[:, 2] == 3))[0]
        filtered_events = stages_events[event_indices]
        
        onsets = []
        durations = []
        descriptions = []
        for event in filtered_events:
            start_time = event[0] / raw.info['sfreq']  # Convert to seconds
            for i in range(6):  # 6 epochs of 5 seconds each
                onsets.append(start_time + i * 5)
                durations.append(5)
                descriptions.append("NREM_epoch")
                
        orig_time = raw.info['meas_date']
        # Create Annotations object
        annotations_nrem_epoch = mne.Annotations(onsets, durations, descriptions, orig_time)
        raw.set_annotations(annotations_nrem_epoch)
        events, event_id = mne.events_from_annotations(raw)
    
        # Write events to a file
        file_name = f's{subject}_{task}_r{run}_nrem'
        filename_events = op.join(bids_path.directory, file_name + '-eve.fif')
        mne.write_events(filename_events, events, overwrite=True)
        
        raw_downsampled = raw.copy().resample(sfreq = resample_sfreq)
        
        #events_downsampled = mne.find_events(raw_downsampled)
        events_downsampled, event_id = mne.events_from_annotations(raw_downsampled)

        raw_downsampled.notch_filter(freqs=notch_filter_freqs)
        raw_downsampled.filter(l_freq=l_freq, h_freq=lowpass_freq)
    
        raw_list.append(raw_downsampled)
        events_list.append(events_downsampled)
        
        #raw_downsampled.plot(start=0, duration = 30,block=True)
    
    
    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list
    raw.plot(start=0, duration = 30,block=False)
    
    
    #%% nrem
    events, events_id = mne.events_from_annotations(raw) #, event_id=events_dict
    events_picks_id = {k:v for k, v in events_id.items() if k.startswith('NREM_epoch')}
    
    # Make epochs 
    epochs = mne.Epochs(raw,
        events, events_picks_id,
        tmin=0 , tmax=5,
        baseline=None,
        proj=False,
        picks = 'all',
        detrend = 0,
        reject=None,
        reject_by_annotation=True, ## need to check if HPI bad epochs were save
        event_repeated = 'merge',
        preload=True,
        verbose=True)
    
    # visually inspect
    evoked = epochs['NREM_epoch'].average()
    evoked.plot()
    plt.show()
    
    
    #%% use autoreject to reject trials 
    #ar = AutoReject()
    #epochs_clean = ar.fit_transform(epochs)
    subset_size = 100  # Number of epochs to use for threshold estimation
    random_indices = np.random.choice(len(epochs), size=subset_size, replace=False)
    epochs_subset = epochs[random_indices]
    
    reject    = get_rejection_threshold(epochs_subset, ch_types=['mag', 'grad'])
    MEGreject = reject#{k: reject[k] for k in list(reject)[:2]}
    del epochs 
    del epochs_subset
    print(MEGreject)
    
    #%% create epochs to calculate filter
    epochs = mne.Epochs(raw,
            events, events_picks_id,
            tmin=0 , tmax=5,
            baseline=None,
            proj=False,
            picks = 'all',
            detrend = 0,
            reject=MEGreject,
            reject_by_annotation=False,
            event_repeated = 'merge',
            preload=True,
            verbose=True)
    
    epochs.plot_drop_log()
    print(deriv_file_nrem)
    epochs.save(deriv_fname_nrem, overwrite=True)
    