import numpy as np
import os
import mne
from mne.preprocessing import EOGRegression
import pandas as pd
from autoreject import get_rejection_threshold


# Define file path
file_path = r"Y:\Projects\2023_Scholte_FMG1441\Data"
SubMatrix_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project"
# Specify the subject name
subject_name = input("Enther the subject number (in form of sub_x): ")
# Read the Image list
Sub_Images_list = pd.read_csv(os.path.join(SubMatrix_path, "Sub_matrix", subject_name + "_randomized_matrix_702.csv"), header=None)
Sub_Images_list = Sub_Images_list.values.tolist()
Sub_Images_list = [i for sub in Sub_Images_list for i in sub]   #Flatten the list
# Read the Image dictionary
All_Images_df = pd.read_csv(os.path.join(SubMatrix_path, "EventsID_Dictionary.csv"), header = None)
All_Images_dict = dict(zip(All_Images_df[0], All_Images_df[1]))
# All_Images_list = pd.read_csv(os.path.join(SubMatrix_path, "All Images' filenames.csv"), header = None)
# All_Images_list = All_Images_list.values.tolist()
# All_Images_list = [i for sub in All_Images_list for i in sub]   #Flatten the list
# values_list = list(range(1, len(All_Images_list)+1))
# All_Images_dict = dict(zip(All_Images_list, values_list))
# df = pd.DataFrame(All_Images_dict.items())
# df.to_csv(os.path.join(SubMatrix_path, "EventsID_Dictionary.csv"), header = None, index = False)
# Read raw data
raw = mne.io.read_raw_bdf(os.path.join(file_path, subject_name, subject_name + ".bdf"))

# Crop data to save memory if needed
# raw.crop(tmax=800)

# Drop irrelevant channels
raw.drop_channels(['EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'])
# raw.rename_channels({'left-ref':'EXG7', 'right-ref':'EXG8'})

# Reset reference channels type from eeg to eog
raw.set_channel_types({'left': 'eog', 'right': 'eog', 'above': 'eog', 'below': 'eog'})
print(raw.info)

# Load data
raw.pick(['eeg', 'eog', 'stim']).load_data()

#Rereference to earlobes/mastoids (average of left and right)
#Check eeg config file the name for mastoids, and then rename them
raw.set_eeg_reference(ref_channels=['left-ref', 'right-ref'])

#What is our baseline?
Baseline = (-0.099609375, 0)

# Is it okay we use the common location map? We have different placement of P5 and P6.
biosemi64_montage = mne.channels.make_standard_montage(kind = 'biosemi64')
EEG_64channels = biosemi64_montage.ch_names
eog_channels = ['left', 'right', 'above', 'below']
# biosemi64_montage.plot()
raw.set_montage(biosemi64_montage, on_missing = 'ignore')
events = mne.find_events(raw)

# Define graphs parameters
plot_kwargs = dict(ylim=dict(eeg=(-15, 15), eog=(-15, 15), csd = (-15,15)))


####################################################################################################################
# Filters:
#     Butterworth Zero Phase Filters
#     Low Cutoff: 0.1 Hz, Time constant 1.5915s, 12 dB/oct ???????????
#     High Cutoff: 30.0 Hz, 24 dB/oct ????????????
#     Notch Filter: 50 Hz and 60 Hz
raw.filter(l_freq=0.1, h_freq=30, method='iir', iir_params=None,picks = 'all')
raw.filter(l_freq=0.1, h_freq=30, method='fir', phase = 'zero', picks = 'all')
# raw.filter(l_freq=0.1, h_freq=30, method='fir', phase = 'zero', filter_length = '1s', picks = 'all')
# alueError: The requested filter length 1025 is too short for the requested 0.05 Hz transition band, which requires 33793 samples
raw.notch_filter(freqs = (50, 60), picks = 'all')
# plot_kwargs = dict(ylim=dict(eeg=(-10, 10), eog=(-5, 15)))
# raw.plot_psd(fmax = 35)
####################################################################################################################


####################################################################################################################
# artifact removal
    # Minimal allowed amplitude: -250.00 µV, Max: 250.00 µV
    # MNE can reject epochs based on maximum peak-to-peak signal amplitude (PTP),
    # i.e. the absolute difference between the lowest and the highest signal value.
    # How can I remove signals based on singel peak but not differences between peaks?

    # Annotating continuous data
    # onsets = events[:, 0] / raw.info['sfreq'] - 0.5
    # durations = [1] * len(events)
    # descriptions = ['bad_event'] * len(events)
    # event_annotation = mne.Annotations(onsets, durations, descriptions, orig_time=raw.info['meas_date'])
    # raw.set_annotations(event_annotation)
####################################################################################################################


####################################################################################################################
# Segmentation
# o Segment size and position relative to reference markers:
# o Start: -100 ms ????, End: 500.00 ms?????, Length: 600 ms?????
# How long the tmin and tmax should be? Images are contiguous to the gray screen
reject_criteria = dict(eeg=50e-6)
flat_criteria = dict(eeg = 50e-8)
events_len = len(events)
CurIndex_event = 0
converted_events = []
for i in range(0,events_len):
    converted_events.append(events[i, :])
    if events[i, 2] > 250:
        converted_events[i][2] = 5000 + events[i, 2]
        continue
    elif 1 <= events[i, 2] <= 250:
        CurImage = Sub_Images_list[CurIndex_event]
        if CurImage.startswith('Stimuli\\Targets\\'):
            CurImage = CurImage[16:]
        elif CurImage.startswith('Stimuli\\'):
            CurImage = CurImage[8:]
        converted_events[i][2] = All_Images_dict[CurImage]
        CurIndex_event += 1
converted_events = np.array(converted_events)
stimuli_list = [str(i) for i in list(range(1,4708))]

#EEG & EOG plots (without reject criteria)
epochs_NoRestriction = mne.Epochs(raw, events, tmin=-0.1, tmax=0.4, picks = EEG_64channels+eog_channels)
epochs_NoRestriction.load_data()
stimuli_epochs_NoRestriction = epochs_NoRestriction[stimuli_list]
stimuli_epochs_NoRestriction.load_data()
stimuli_epochs_plot_before_NoRestriction = stimuli_epochs_NoRestriction.average('eeg').plot(**plot_kwargs)
stimuli_epochs_plot_before_NoRestriction.set_size_inches(27, 16)
stimuli_epochs_plot_before_NoRestriction.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuliEEG_epochs_before(All events).png"))
stimuli_epochs_plot_before_NoRestriction = stimuli_epochs_NoRestriction.average('eog').plot(**plot_kwargs)
stimuli_epochs_plot_before_NoRestriction.set_size_inches(27, 16)
stimuli_epochs_plot_before_NoRestriction.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuliEOG_epochs_before(All events).png"))

#EEG & EOG plots (Use  AutoReject criteria)
AutoReject = get_rejection_threshold(epochs_NoRestriction)
epochs_autoreject = mne.Epochs(raw, events, tmin=-0.1, tmax=0.4, baseline = Baseline, reject = AutoReject, picks = EEG_64channels+eog_channels)
stimuli_epochs_AutoReject = epochs_autoreject[stimuli_list]
stimuli_epochs_AutoReject.load_data()
stimuli_epochs_plot_before_AutoReject = stimuli_epochs_AutoReject.average('eeg').plot(**plot_kwargs)
stimuli_epochs_plot_before_AutoReject.set_size_inches(27, 16)
stimuli_epochs_plot_before_AutoReject.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuliEEG_epochs_before(AutoReject).png"))
stimuli_epochs_plot_before_AutoReject = stimuli_epochs_AutoReject.average('eog').plot(**plot_kwargs)
stimuli_epochs_plot_before_AutoReject.set_size_inches(27, 16)
stimuli_epochs_plot_before_AutoReject.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuliEOG_epochs_before(AutoReject).png"))

#EEG & EOG plots (Use self-defined reject criteria)
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.4, baseline = Baseline, reject = reject_criteria, flat = flat_criteria, picks = EEG_64channels+eog_channels)
epochs.load_data()
stimuli_epochs = epochs[stimuli_list]
stimuli_epochs.load_data()
stimuli_epochs_plot_before = stimuli_epochs.average('eeg').plot(**plot_kwargs)
stimuli_epochs_plot_before.set_size_inches(27, 16)
stimuli_epochs_plot_before.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochsEEG_before(with reject criteria).png"))
stimuli_epochs_plot_before = stimuli_epochs.average('eog').plot(**plot_kwargs)
stimuli_epochs_plot_before.set_size_inches(27, 16)
stimuli_epochs_plot_before.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochsEOG_before(with reject criteria).png"))
####################################################################################################################


####################################################################################################################
# Ocular correction (Gratton & Coles)
model_plain = EOGRegression(picks='eeg', picks_artifact='eog').fit(stimuli_epochs)
stimuli_epochs_clean_plain = model_plain.apply(stimuli_epochs)
stimuli_epochs_clean_plain.apply_baseline()
stimuli_epochs_clean_plain.save(os.path.join(SubMatrix_path, subject_name + "-OC-epo.fif"), overwrite = False)
stimuli_epochs_eogout_plot = stimuli_epochs_clean_plain.average().plot(**plot_kwargs)
stimuli_epochs_eogout_plot.set_size_inches(27, 16)
stimuli_epochs_eogout_plot.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochs_after_OC(with reject criteria).png"))

model_plain_AutoReject = EOGRegression(picks='eeg', picks_artifact='eog').fit(stimuli_epochs_AutoReject)
stimuli_epochs_clean_plain_AutoReject = model_plain_AutoReject.apply(stimuli_epochs_AutoReject)
stimuli_epochs_clean_plain_AutoReject.apply_baseline()
stimuli_epochs_clean_plain_AutoReject.save(os.path.join(SubMatrix_path, subject_name + "-OC-AutoReject-epo.fif"), overwrite = False)
stimuli_epochs_eogout_plot_AutoReject = stimuli_epochs_clean_plain_AutoReject.average().plot(**plot_kwargs)
stimuli_epochs_eogout_plot_AutoReject.set_size_inches(27, 16)
stimuli_epochs_eogout_plot_AutoReject.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochs_after_OC(AutoReject).png"))

model_plain_NoRestriction = EOGRegression(picks='eeg', picks_artifact='eog').fit(stimuli_epochs_NoRestriction)
stimuli_epochs_clean_plain_NoRestriction = model_plain_NoRestriction.apply(stimuli_epochs_NoRestriction)
stimuli_epochs_clean_plain_NoRestriction.apply_baseline()
stimuli_epochs_clean_plain_NoRestriction.save(os.path.join(SubMatrix_path, subject_name + "-OC-NoRestriction-epo.fif"), overwrite = False)
stimuli_epochs_eogout_plot_NoRestriction= stimuli_epochs_clean_plain_NoRestriction.average().plot(**plot_kwargs)
stimuli_epochs_eogout_plot_NoRestriction.set_size_inches(27, 16)
stimuli_epochs_eogout_plot_NoRestriction.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochs_after_OC(All events).png"))
####################################################################################################################


####################################################################################################################
# CSD transformation
#     o Order of splines: 4
#     o Maximal degree of Legendre polynomials: 10
#     o Approximation parameter Lambda: 1.000000e-005
stimuli_epochs_csd = mne.preprocessing.compute_current_source_density(stimuli_epochs_clean_plain, lambda2=1e-5, stiffness=4, n_legendre_terms=10)
stimuli_epochs_csd.save(os.path.join(SubMatrix_path,subject_name + "-OC&CSD-epo.fif"), overwrite = False)
evoked_plot = stimuli_epochs_csd.average().plot(**plot_kwargs)
evoked_plot.set_size_inches(27, 16)
evoked_plot.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochs_after_OC&CSD(with reject criteria).png"))

stimuli_epochs_csd_AutoReject = mne.preprocessing.compute_current_source_density(stimuli_epochs_clean_plain_AutoReject, lambda2=1e-5, stiffness=4, n_legendre_terms=10)
stimuli_epochs_csd_AutoReject.save(os.path.join(SubMatrix_path,subject_name + "-OC&CSD-AutoReject-epo.fif"), overwrite = False)
evoked_plot_AutoReject = stimuli_epochs_csd_AutoReject.average().plot(**plot_kwargs)
evoked_plot_AutoReject.set_size_inches(27, 16)
evoked_plot_AutoReject.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochs_after_OC&CSD(AutoReject).png"))

stimuli_epochs_csd_NoRestriction = mne.preprocessing.compute_current_source_density(stimuli_epochs_clean_plain_NoRestriction, lambda2=1e-5, stiffness=4, n_legendre_terms=10)
stimuli_epochs_csd_NoRestriction.save(os.path.join(SubMatrix_path,subject_name + "-OC&CSD-NoRestriction-epo.fif"), overwrite = False)
evoked_plot_NoRestriction = stimuli_epochs_csd_NoRestriction.average().plot(**plot_kwargs)
evoked_plot_NoRestriction.set_size_inches(27, 16)
evoked_plot_NoRestriction.savefig(os.path.join(SubMatrix_path,subject_name + "_stimuli_epochs_after_OC&CSD(All events).png"))
####################################################################################################################



####################################################################################################################


