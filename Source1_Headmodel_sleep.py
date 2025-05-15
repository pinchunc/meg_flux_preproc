import os.path as op
import os
import mne
import matplotlib
import nibabel

from mne_bids import BIDSPath, read_raw_bids
# import settings
from FLUXSettings import bids_root,subject,task_list,session
bids_root = '/Volumes/MEG_BIDS/BIDS-data/'
subject = '104'

#get_ipython().run_line_magic("matplotlib", "qt")
matplotlib.use('Qt5Agg')

task = 'sleep'
meg_suffix = 'meg'
ann_suffix = 'ann'
mri_suffix = 'T1w'
epo_suffix = 'epo'
bem_suffix = 'bem-sol'
src_suffix = 'src'
fwd_suffix = 'fwd'
trans_suffix = 'trans'

deriv_root   = op.join(bids_root, 'derivatives/analysis')
preproc_root   = op.join(bids_root, 'derivatives/preprocessing')

fs_subject = 'sub-' + subject  # name of subject from freesurfer

# Files and directories for input
bids_path = BIDSPath(subject=subject, session=session,
            task=task, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)

# only load the first run for head position
# raw_path = bids_path.copy().update(root=bids_root, 
#            suffix=meg_suffix, extension='.fif', run='01', check=False) 
ann_path = bids_path.copy().update(root=preproc_root, 
           suffix=ann_suffix, extension='.fif', run='01', check=False) 

print("* Annotation fiff-file (with digitization points) :", ann_path)
print("* Epoched file: ", bids_path)

mri_root = BIDSPath(subject=subject, session=session,
                    root=bids_root,  
                    datatype='anat', suffix=mri_suffix, extension='.nii')
print("* MRI file: ",mri_root) 

fs_root = op.join(bids_root, 'derivatives', 'FreeSurfer')
print("* Freesurfer directory: ",fs_root)

# Files and directories for generated output 
print("\n*** Output files ***")

fwd_fname = bids_path.basename.replace(epo_suffix, fwd_suffix)
fwd_file = op.join(bids_path.directory, fwd_fname)
print("* Forward model: ",fwd_file)

src_file = fwd_file.replace(fwd_suffix, src_suffix)
print("* Brain surface file: ",src_file)

trans_file = fwd_file.replace(fwd_suffix, trans_suffix)
print("* Tranformation file: ",trans_file)

bem_file = fwd_file.replace(fwd_suffix, bem_suffix)
print("* Boundary element file: ",bem_file)

#%%
# read MRI
t1_fname = f'{fs_root}/s{subject}/mri/T1.mgz'
t1 = nibabel.load(t1_fname)
t1.orthoview()
print(t1.affine)
#%%
## check parcellation from FreeSurfer
Brain = mne.viz.get_brain_class()

brain = Brain(f's{subject}', 
              hemi='lh', 
              surf='pial',
              subjects_dir=fs_root, 
              size=(800, 600))

brain.add_annotation('aparc.a2009s', borders=False)

#%% BEM
conductivity = (0.3,) 
model = mne.make_bem_model(f's{subject}',
                           ico=4,
                           conductivity=conductivity,
                           subjects_dir=fs_root)
bem = mne.make_bem_solution(model)

mne.write_bem_solution(bem_file,
                       bem, overwrite=True)
#%% visualize bem
mne.viz.plot_bem(subject=f's{subject}',
                subjects_dir=fs_root,
                brain_surfaces='white',
                orientation='coronal')

#%% Co-registration with anatomical landmars 
print("* Tranformation file: ",trans_file)
mne.gui.coregistration(subject=f's{subject}', subjects_dir=fs_root,block=True)

#%% Visualize coregistration
# Replace with the path to your epochs file
info = read_raw_bids(bids_path=ann_path, verbose=False).info
#info = read_raw_bids(bids_path=raw_path, verbose=False).info
print(info)
mne.viz.plot_alignment(info, trans=trans_file, subject=f's{subject}', dig=True,
                           meg=['helmet', 'sensors'], subjects_dir=fs_root,
                           surfaces='head-dense')

#%% Computing the sources according to the BEM model
surface = op.join(fs_root, f's{subject}', 'bem', 'inner_skull.surf')
src = mne.setup_volume_source_space(f's{subject}', subjects_dir=fs_root,
                                     surface=surface,
                                     verbose=True)
mne.write_source_spaces(src_file, src, overwrite=True)

#%% Visulize BEM
mne.viz.plot_bem(subject=f's{subject}', 
                     subjects_dir=fs_root,
                     brain_surfaces='white', 
                     src=src, 
                     orientation='coronal')
#%% Construting the forward model
import mne

fwd = mne.make_forward_solution(info, 
                                trans=trans_file,
                                src=src, 
                                bem=bem,
                                meg=True, 
                                eeg=False, 
                                mindist=5.,  #TODO: minimum distance of sources from inner skull surface (in mm); can be 2.5
                                n_jobs=-1, 
                                verbose=True)
mne.write_forward_solution(fwd_file, fwd, overwrite=True)
