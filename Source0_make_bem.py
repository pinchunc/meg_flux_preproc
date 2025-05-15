# make sure to conda activate osl
from mne import bem
import os.path as op

subject_dir= '/Applications/freesurfer/7.3.2/subjects'

sub_list = ['s111', 's116', 's117', 's119']#'s102', 's103', 's104', 's106', 's112', 's118',
for subject in sub_list:
    bem.make_scalp_surfaces(subject, subject_dir, overwrite=True)
    bem.make_watershed_bem(subject, subjects_dir=subject_dir, overwrite=True)