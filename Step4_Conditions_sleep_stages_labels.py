"""
this code add sleep stages label to the ica.fif file and pad edge epochs as artifacts
"""
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
#import settings
get_ipython().run_line_magic("matplotlib", "qt")

bids_root = '/Volumes/MEG_BIDS/BIDS-data/'

resample_sfreq = 500
l_freq = 0.35
h_freq = 200 # for gamma
iir_params = dict(order=5, ftype='butter')
notch_filter_freqs = (50, 100, 150) # must be lowe than lowpass
task = 'sleep'
session = '01'
bids_root  = '/Volumes/MEG_BIDS/BIDS-data'# '/data/xpsy-memori/xpsy1260/megmori/BIDS-data'
meg_suffix = 'meg'
ica_suffix = 'ica'
epo_suffix = 'epo'
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')
sub_list = ['118']#'102', '103', '104', '106', '107', '112', '118', '110', '111','116', '117', '119'

def is_overlap(annot1, annot2): # add a 250ms buffer before and after
    annot1_end = annot1['onset'] + annot1['duration'] + 0.25
    annot2_end = annot2['onset'] + annot2['duration'] + 0.25
    return not (annot1['onset'] - 0.25 >= annot2_end or annot2['onset'] - 0.25 >= annot1_end)

for subject in sub_list:
    bids_path = BIDSPath(subject=subject, session=session,
                task=task, run='01', suffix=epo_suffix, datatype='meg',
                root=deriv_root, extension='.fif', check=False).mkdir()
    
    deriv_file_nrem = bids_path.basename.replace('run-01', 'run-all-nrem-5sec')  # run 12 -> run 01 concatenated with run 02
    deriv_fname_nrem = op.join(bids_path.directory, deriv_file_nrem)
    
    #%% continuous epochs of nrem  per sleep scan and add annotations of stages and artifacts
    task = 'sleep'
    files = os.listdir(os.path.join(preproc_root, f'sub-{subject}', f'ses-{session}', 'meg'))
    task_files = [file for file in files if file.endswith('ica.fif') and file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]
    
    for run_idx in [6,7]:#range(len(task_files)):
        # start from run3
        run = run_idx+1
        run = f"{run:02}"
        bids_path_preproc = BIDSPath(subject=subject, session=session,
                                     task=task, run=run, suffix=meg_suffix, datatype='meg',
                                     root=preproc_root, extension='.fif', check=False)
    
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, suffix=epo_suffix, datatype='meg',
                    root=deriv_root, extension='.fif', check=False)
        deriv_file_nrem = bids_path.basename.replace('task-sleep', 'task-sleep-staged')
        deriv_file_nrem = deriv_file_nrem.replace('epo', 'ica') 
        
        deriv_fname_nrem = op.join(bids_path.directory, deriv_file_nrem)
        print(bids_path_preproc.fpath)
        
        # Create event markers
        ica_path = BIDSPath(subject=subject, session=session,
                            task=task, run=run, suffix=ica_suffix, datatype='meg',
                            root=preproc_root, extension='.fif', check=False).mkdir()
        raw = mne.io.read_raw_fif(ica_path, allow_maxshield=True, verbose=True, preload=False)
        #events_bad_muscle = mne.events_from_annotations(raw, regexp = 'BAD_muscle')
        #events_bad_hpi = mne.events_from_annotations(raw, regexp = 'BAD_HPI')
        
        onsets = []
        durations = []
        descriptions = []    
        for ann in raw.annotations:
            if ann["description"] == 'BAD_HPI' or ann["description"] == 'BAD_muscle':
                descriptions.append(ann["description"])
                onsets.append(ann["onset"])
                durations.append(ann["duration"])
                
        orig_time = raw.info['meas_date']
        # Create Annotations object
        annotations_BAD = mne.Annotations(onsets, durations, descriptions, orig_time)
        
        
        sfreq = raw.info['sfreq']  # should be 1000 Hz in your case    
        decim = np.round(sfreq / resample_sfreq).astype(int)
        obtained_sfreq = sfreq / decim
        lowpass_freq = obtained_sfreq / 3.0
        
        # load sleep stages
        stage_fname = op.join(bids_root, f'sleep_edf/s{subject}/auto_stage_s{subject}_r{run_idx+1}.mat')
        stage_data = scipy.io.loadmat(stage_fname)
        stage_data = stage_data['stageData']['stages'][0][0]
        
        # Create event markers
        epoch_duration = 30  # seconds
        stages_events = []
        for i, stage in enumerate(stage_data):
            start_sample = int(i * epoch_duration * sfreq) + raw.first_samp
            stages_events.append([start_sample, 0, stage[0]])
        
        stages_events = np.array(stages_events)
        
        onsets = []
        durations = []
        descriptions = []
        for event in stages_events:
            start_time = event[0] / raw.info['sfreq']  # Convert to seconds
            for i in range(30):  # 6 epochs of 5 seconds each
                onsets.append(start_time + i)
                durations.append(1)
                descriptions.append(f'stage{event[2]}')
                
        orig_time = raw.info['meas_date']
        # Create Annotations object
        annotations_stages = mne.Annotations(onsets, durations, descriptions, orig_time)
        
        ## replace stages with artifact labels
        # Initialize lists to store the new annotations
        new_onsets = []
        new_durations = []
        new_descriptions = []
        
        # Loop over each annotation in annotations_stages
        for annot2 in annotations_stages:
            updated = False
            for annot1 in annotations_BAD:
                if is_overlap(annot1, annot2):
                    # Redefine annotations as needed, here we use annot2 description
                    # Validation to ensure duration is not negative
                    if annot2['duration'] >= 0:
                        new_onsets.append(annot2['onset'])
                        new_durations.append(annot2['duration'])
                        new_descriptions.append('artifact')
                    else:
                        print(f"Skipped negative duration annotation: Onset = {annot2['onset']}, Duration = {annot2['duration']}, Description = {annot2['description']}")
                    updated = True
                    break
            if not updated:
                # If no overlap was found, keep the original annotation
                if annot2['duration'] >= 0:
                    new_onsets.append(annot2['onset'])
                    new_durations.append(annot2['duration'])
                    new_descriptions.append(annot2['description'])
                else:
                    print(f"Skipped negative duration annotation: Onset = {annot2['onset']}, Duration = {annot2['duration']}, Description = {annot2['description']}")
        
        # Fill in the end of the run with stage labels
        total_duration = raw.times[-1] + raw.first_samp / raw.info['sfreq']
        if new_onsets and new_onsets[-1] < total_duration:
            duration_to_add = total_duration - new_onsets[-1] - 1
            if duration_to_add > 0:
                new_durations.append(duration_to_add)
                new_onsets.append(new_onsets[-1] + 1)
                new_descriptions.append('artifact')
            else:
                print("Warning: Skipped adding end-of-run annotation due to negative or zero duration.")

        new_annotations_stages = mne.Annotations(onset=new_onsets, duration=new_durations, description=new_descriptions, orig_time=orig_time)
    
        raw.set_annotations(new_annotations_stages + annotations_BAD)
        events_dict = {'stage0': 1, 'stage1': 2, 'stage2': 3, 'stage3': 4, 'stage5': 5, 'artifact': 6, 'stage6': 7}
        events, event_id = mne.events_from_annotations(raw, event_id=events_dict)
    
        
        raw_downsampled = raw.copy().resample(sfreq = resample_sfreq)
        events_downsampled, event_id_downsampled = mne.events_from_annotations(raw_downsampled, event_id=events_dict)
        raw_downsampled.plot(start = 0, duration = 30)#, block = True)
            
        # Write events to a file
        file_name = deriv_fname_nrem.replace('_ica','') #f's{subject}_{task}_r{run}_staged'
        file_name = file_name.replace('.fif','')
        filename_events = op.join(bids_path.directory, file_name + '-eve.fif')
        mne.write_events(filename_events, events_downsampled, overwrite=True)
        
        raw_downsampled.notch_filter(freqs=notch_filter_freqs)
        raw_downsampled.filter(l_freq=l_freq, h_freq=lowpass_freq)
        
        
        ## pause here
        raw_downsampled.save(deriv_fname_nrem, overwrite=True)
        