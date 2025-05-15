import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids
from autoreject import get_rejection_threshold
import scipy.io
from FLUXSettings import bids_root, subject, task_list, session

# Set matplotlib to use the 'qt' backend
get_ipython().run_line_magic("matplotlib", "qt")

# Set paths and parameters
bids_root = '/Volumes/MEG_BIDS/BIDS-data/'
#subject = '112'
session = '01'
task = 'sleep'
resample_sfreq = 500
l_freq = 0.35
iir_params = dict(order=5, ftype='butter')
notch_filter_freqs = (50,)
this_scalp_chan = 'Cz'
meg_suffix = 'meg'
ica_suffix = 'ica'
epo_suffix = 'epo'
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root = op.join(bids_root, 'derivatives/analysis')

#%%
task = 'sleep'
sub_list = ['102','103', '104', '106','107', '112', '118', '110','111','116', '117', '119']
# 
for subject in sub_list:
    # Load sleep events timing
    slythm_fname = op.join(deriv_root, f'sub-{subject}', f'ses-{session}', 'megeeg', 'megeeg_slythms_NREM_Cz.mat')
    slythm_fname = slythm_fname.replace('Cz', this_scalp_chan)
    slythm_data = scipy.io.loadmat(slythm_fname)
    
    # Extract spindle and SO event data
    spindles_minTime_timestamps = slythm_data['spindles_minTime_timestamps'][0]
    spindles_minTime_run_idx = slythm_data['spindles_minTime_run_idx'][0]
    spindles_surrogate_minTime_timestamps = slythm_data['spindles_surrogate_minTime_timestamps'][0]
    spindles_surrogate_minTime_run_idx = slythm_data['spindles_surrogate_minTime_run_idx'][0]
    
    SO_minTime_timestamps = slythm_data['SO_minTime_timestamps'][0]
    SO_minTime_run_idx = slythm_data['SO_minTime_run_idx'][0]
    SO_surrogate_minTime_timestamps = slythm_data['SO_surrogate_minTime_timestamps'][0]
    SO_surrogate_minTime_run_idx = slythm_data['SO_surrogate_minTime_run_idx'][0]
    
    # Prepare BIDS path
    bids_path = BIDSPath(subject=subject, session=session, task=task, run=1, suffix=epo_suffix, datatype='meg',
                         root=deriv_root, extension='.fif', check=False).mkdir()
    
    deriv_fname_spindles = op.join(bids_path.directory, bids_path.basename.replace('run-01', 'run-all-spindles-Cz')).replace('Cz', this_scalp_chan)
    deriv_fname_SO = op.join(bids_path.directory, bids_path.basename.replace('run-01', 'run-all-SO-Cz')).replace('Cz', this_scalp_chan)
    
    # Process each run
    raw_list = []
    events_list = []
    
    files = os.listdir(op.join(deriv_root,f'sub-{subject}',f'ses-{session}','meg'))
    task_files = [file for file in files if file.endswith('ica.fif') & file.startswith(f'sub-{subject}_ses-{session}_task-{task}-staged')]
    
    for run_idx, task_file in enumerate(task_files):
        run = run_idx + 1
        ica_path = BIDSPath(subject=subject, session=session, task=task, run=f"{run:02}", suffix=ica_suffix, datatype='meg',
                            root=deriv_root, extension='.fif', check=False).mkdir()
        task_file_fname = op.join(ica_path.directory, task_file)
    
        raw = mne.io.read_raw_fif(task_file_fname, allow_maxshield=True, verbose=True, preload=True)
        raw.resample(sfreq=resample_sfreq)
    
        # Decimate timestamps
        sfreq = raw.info['sfreq']
        decim = int(np.round(sfreq / resample_sfreq))
        lowpass_freq = sfreq / (3 * decim)
    
        run_spindles_minTime_timestamps = np.rint(spindles_minTime_timestamps[spindles_minTime_run_idx == run])
        run_spindles_surrogate_minTime_timestamps = np.rint(spindles_surrogate_minTime_timestamps[spindles_surrogate_minTime_run_idx == run])
        run_SO_minTime_timestamps = np.rint(SO_minTime_timestamps[SO_minTime_run_idx == run])
        run_SO_surrogate_minTime_timestamps = np.rint(SO_surrogate_minTime_timestamps[SO_surrogate_minTime_run_idx == run])
    
        event_val_spindles = 15
        event_val_spindles_surr = 14
        event_val_SO = 20
        event_val_SO_surr = 19
        
        # Load stage data and check if stages 2 or 3 are present
        stage_fname = op.join(bids_root, f'sleep_edf/s{subject}/auto_stage_s{subject}_r{run}.mat')
        stage_data = scipy.io.loadmat(stage_fname)['stageData']['stages'][0][0]
    
        if not (2 in stage_data or 3 in stage_data):
            print(f"Stages 2 or 3 not found in stage data for run {run}. Skipping...")
        elif (run_spindles_minTime_timestamps.size == 0 or
            run_spindles_surrogate_minTime_timestamps.size == 0 or
            run_SO_minTime_timestamps.size == 0 or
            run_SO_surrogate_minTime_timestamps.size == 0):
            print(f"No spindles or SO events found for run {run}. Skipping...")
        else:
            # Create event markers for stages
            epoch_duration = 30  # seconds
            stages_events = []
            for i, stage in enumerate(stage_data):
                start_sample = int(i * epoch_duration * sfreq) + raw.first_samp
                stages_events.append([start_sample, 0, stage[0]])
            
            stages_events = np.array(stages_events)
            onset             = stages_events[:, 0] / raw.info['sfreq']
            n_stages          = len(stages_events)
            duration          = np.repeat(epoch_duration, n_stages)
            description       = [str(stage) for stage in stages_events[:, 2]]
            orig_time         = raw.info['meas_date']
            annotations_stages = mne.Annotations(onset, duration, description, orig_time)
        
            
            #
            spindles_events = []
            for count, spindles in enumerate(run_spindles_minTime_timestamps):    
                start_sample = spindles  + raw.first_samp
                spindles_events.append([start_sample, 0, event_val_spindles])
            
            spindles_events   = np.array(spindles_events)
            onset             = spindles_events[:, 0] / raw.info['sfreq']
            n_spindles        = len(spindles_events)
            duration          = np.repeat(0, n_spindles)
            description       = ['spindles_events'] * n_spindles
            orig_time         = raw.info['meas_date']
            annotations_spindles   = mne.Annotations(onset, duration, description, orig_time)
            
            #
            spindles_surr_events = []
            for count, spindles in enumerate(run_spindles_surrogate_minTime_timestamps):    
                start_sample = spindles  + raw.first_samp
                spindles_surr_events.append([start_sample, 0, event_val_spindles_surr])
            
            spindles_surr_events   = np.array(spindles_surr_events)
            onset             = spindles_surr_events[:, 0] / raw.info['sfreq']
            n_spindles_surr   = len(spindles_surr_events)
            duration          = np.repeat(0, n_spindles_surr)
            description       = ['spindles_surr_events'] * n_spindles_surr
            orig_time         = raw.info['meas_date']
            annotations_spindles_surr   = mne.Annotations(onset, duration, description, orig_time)
            
            #
            SO_events = []
            for count, SO in enumerate(run_SO_minTime_timestamps):    
                start_sample = SO + raw.first_samp
                SO_events.append([start_sample, 0, event_val_SO])
            
            SO_events         = np.array(SO_events)
            onset             = SO_events[:, 0] / raw.info['sfreq']
            n_SO              = len(SO_events)
            duration          = np.repeat(0, n_SO)
            description       = ['SO_events'] * n_SO
            orig_time         = raw.info['meas_date']
            annotations_SO    = mne.Annotations(onset, duration, description, orig_time)
            
            SO_surr_events = []
            for count, SO in enumerate(run_SO_surrogate_minTime_timestamps):    
                start_sample = SO + raw.first_samp
                SO_surr_events.append([start_sample, 0, event_val_SO_surr])
            
            SO_surr_events    = np.array(SO_surr_events)
            onset             = SO_surr_events[:, 0] / raw.info['sfreq']
            n_SO_surr         = len(SO_surr_events)
            duration          = np.repeat(0, n_SO_surr)
            description       = ['SO_surr_events'] * n_SO_surr
            orig_time         = raw.info['meas_date']
            annotations_SO_surr    = mne.Annotations(onset, duration, description, orig_time)
            
            # Set annotations to raw data
            raw.set_annotations(annotations_spindles + annotations_spindles_surr + annotations_SO + annotations_SO_surr + annotations_stages)
            #raw.plot(start=0, duration=30.0, block=True)
        
            # Convert annotations to events
            events, event_id = mne.events_from_annotations(raw)
            mne.write_events(op.join(bids_path.directory, f's{subject}_{task}_r{run}_slythm_NREM_{this_scalp_chan}-eve.fif'), events, overwrite=True)
        
            # Filter and store raw data
            raw.filter(l_freq=l_freq, h_freq=lowpass_freq)
            raw_list.append(raw)
            events_list.append(events)
    
    # Concatenate raw data from all runs
    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list
    
    #%% spindles_events
    #events_dict = {'Body': 1, 'Face': 2, 'Object': 3, 'Scene': 4, 'Word': 5, 'Control': 6, 'Repeated': 999}
    events, events_id = mne.events_from_annotations(raw) #, event_id=events_dict
    events_picks_id = {k:v for k, v in events_id.items() if k.startswith('spindles_events') or k.startswith('spindles_surr_events')}
    
    # Make epochs 
    epochs = mne.Epochs(raw,
        events, events_picks_id,
        tmin=-4 , tmax=4,
        baseline=None,
        proj=False,
        picks = 'all',
        detrend = 0,
        reject=None,
        reject_by_annotation=True, ## need to check if HPI bad epochs were save
        preload=True,
        verbose=True)
    
    # visually inspect
    evoked = epochs['spindles_events'].average()
    evoked.plot()
    plt.show()
    
    
    #%% use autoreject to reject trials 
    #ar = AutoReject()
    #epochs_clean = ar.fit_transform(epochs)
    subset_size = 200  # Number of epochs to use for threshold estimation
    # Adjust subset_size to avoid the ValueError
    if len(epochs) <= subset_size:
        random_indices = np.arange(len(epochs))  # Use all epochs
    else:
        random_indices = np.random.choice(len(epochs), size=subset_size, replace=False)
    epochs_subset = epochs[random_indices]
    
    reject    = get_rejection_threshold(epochs_subset, ch_types=['mag', 'grad'])
    MEGreject = reject#{k: reject[k] for k in list(reject)[:2]}
    del epochs 
    del epochs_subset
    print(MEGreject)
    
    #%%
    epochs = mne.Epochs(raw,
            events, events_picks_id,
            tmin=-4 , tmax=4,
            baseline=None,
            proj=False,
            picks = 'all',
            detrend = 0,
            reject=MEGreject,
            reject_by_annotation=False,
            preload=True,
            verbose=True)
    
    epochs.plot_drop_log()
    print(deriv_fname_spindles)
    epochs.save(deriv_fname_spindles, overwrite=True)
    
    # #%%  SO_events
    # events, events_id = mne.events_from_annotations(raw) #, event_id=events_dict
    # events_picks_id = {k:v for k, v in events_id.items() if k.startswith('SO_events') or k.startswith('SO_surr_events')}
    
    # # Make epochs 
    # epochs = mne.Epochs(raw,
    #     events, events_picks_id,
    #     tmin=-4 , tmax=4,
    #     baseline=None,
    #     proj=False,
    #     picks = 'all',
    #     detrend = 0,
    #     reject=None,
    #     reject_by_annotation=True, ## need to check if HPI bad epochs were save
    #     preload=True,
    #     verbose=True)
    
    # # visually inspect
    # evoked = epochs['SO_events'].average()
    # evoked.plot()
    # plt.show()
    
    # #%% use autoreject to reject trials 
    # #ar = AutoReject()
    # #epochs_clean = ar.fit_transform(epochs)
    # subset_size = 100  # Number of epochs to use for threshold estimation
    # # Adjust subset_size to avoid the ValueError
    # if len(epochs) <= subset_size:
    #     random_indices = np.arange(len(epochs))  # Use all epochs
    # else:
    #     random_indices = np.random.choice(len(epochs), size=subset_size, replace=False)
    
    # epochs_subset = epochs[random_indices]
    # reject    = get_rejection_threshold(epochs_subset, ch_types=['mag', 'grad'])  
    # MEGreject = reject#{k: reject[k] for k in list(reject)[:2]}
    # del epochs
    # print(MEGreject)
    
    # #%%
    # epochs = mne.Epochs(raw,
    #         events, events_picks_id,
    #         tmin=-4 , tmax=4,
    #         baseline=None,
    #         proj=False,
    #         picks = 'all',
    #         detrend = 0,
    #         reject=MEGreject,
    #         reject_by_annotation=False,
    #         preload=True,
    #         verbose=True)
    
    # epochs.plot_drop_log()
    # print(deriv_fname_SO)
    # epochs.save(deriv_fname_SO, overwrite=True)
