import os.path as op
import os
import matplotlib.pyplot as plt
import numpy as np 

import mne
from mne.annotations import Annotations

from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids, 
                      write_meg_calibration, write_meg_crosstalk, 
                      get_anat_landmarks, write_anat)

from FLUXSettings import bids_root,subject,session

bids_root = '/Volumes/MEG_BIDS/BIDS-data/'

subject = '102'
#%%
# specify specific file names
file_extension = '.fif'

# Define the fine calibration and cross-talk compensation files for BIDS
maxfilter_folder = op.join(bids_root, 'crosstalk')

# files from Anna Camera for our MEG system, same for all our recordings
crosstalk_file   = op.join(maxfilter_folder, 'ct_sparse.fif')  #'reduces interference' 
                                                                #'between Elekta's co-located' 
                                                                #'magnetometer and'
                                                                #'paired gradiometer sensor units'
calibration_file = op.join(maxfilter_folder, 'sss_cal.dat')  #'encodes site-specific'
                                                                #'information about sensor' 
                                                                #'orientations and calibration'

#%% MRI bids conversion
## DO THIS AFTER HEADMODEL IS COREGISTERED
deriv_root   = op.join(bids_root, 'derivatives/analysis')
trans_bids_path = BIDSPath(subject=subject, session=session,
                            task='loc', run=12, suffix='trans', datatype='meg',
                            root=deriv_root, extension='.fif', check=False)

fs_sub_dir = f'{bids_root}/derivatives/FreeSurfer'
fs_subject = 'sub-' + subject  # name of subject from freesurfer
t1_fname = op.join(fs_sub_dir, fs_subject, 'mri', 'T1.mgz')

# Create the BIDSpath object
""" creat MRI specific bidspath object and then use trans file to transform 
landmarks from the raw file to the voxel space of the image"""

t1w_bids_path = BIDSPath(subject=subject, session=session, 
                          root=bids_root, suffix='T1w')

meg_bids_path = BIDSPath(subject=subject, session=session,
                      task='loc', run=1, root=bids_root)
info = read_raw_bids(bids_path=meg_bids_path, verbose=False).info

trans = mne.read_trans(trans_bids_path)  
landmarks = get_anat_landmarks(
    image=t1_fname,  # path to the nifti file
    info=info,       # MEG data file info from the subject
    trans=trans,
    fs_subject=fs_subject,
    fs_subjects_dir=fs_sub_dir)

t1w_bids_path = write_anat(
    image=t1_fname, bids_path=t1w_bids_path,
    landmarks=landmarks, deface=False,
    overwrite=True, verbose=True)