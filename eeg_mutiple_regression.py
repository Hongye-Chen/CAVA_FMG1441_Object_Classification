import dill
import numpy as np
import mne
import pandas as pd
import os
import yaml
from sklearn import linear_model

epoch_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project\EEG_Preprocessing\sub_12\sub_12-OC&CSD-AutoReject-epo.fif"
dic_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project\EventsID_Dictionary.csv"
file_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project\Stimuli"
# Read the Image dictionary
All_Images_df = pd.read_csv(dic_path, header = None)
All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))

# Get CE & SC list
with open(os.path.join(file_path, "lgn_statistics.pkl"), 'rb') as f:
    result = dill.load(f)
statistics_filename = result['filenames']
# The first dim: number of images, second: the RGB, third: number of random crops, fourth:  center vs. periphery
ce = result['CE']
ce_list = ce[:,:,0,0]
ce_list = ce_list.mean(axis = 1)  # Average across RGB channels
sc = result['SC']
sc_list = sc[:,:,0,0]
sc_list = sc_list.mean(axis = 1)

# Get Epochs
sub_epochs = mne.read_epochs(epoch_path)
oz_epochs = sub_epochs.pick_channels(ch_names = ['Oz'])
# Get events filename
oz_events = oz_epochs.events[2]
oz_events_filename = [All_Images_dict[i] for i in oz_events]

# Get the subject unique images list
with open(os.path.join(file_path, "eeg_oads_stimulus_filenames.yml"), 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)
subjects = subjects['sub_12']
sub_sti = [i[8:] for i in subjects if len(i) < 37]

# Get index of images in the events
images_index = []
for i in sub_sti:
    images_index.append(oz_events_filename.index(i))

# Use the index to get the first set of 702 events EEG data
oz_702data = oz_epochs._data[images_index,:,:]

# Get SC and CE index and get the corresponding sc and ce list
statistics_index = []
for i in sub_sti:
    statistics_index.append(statistics_filename.index(i))
ce_702 = ce_list[statistics_index]
sc_702 = sc_list[statistics_index]

# Fit linear regression model for each time points
sub_12_r2 = []
x = [ [i, j] for i , j in zip(ce_702, sc_702)]
for i in range(oz_702data.shape[2]):
    y = np.array(oz_702data[:, :, i])
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    sub_12_r2.append(regr.score(x, y))
