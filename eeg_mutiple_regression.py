import dill
import numpy as np
import mne
import pandas as pd
import os
import yaml
from sklearn import linear_model
import matplotlib.pyplot as plt

# Enter subject number
subject_name = input("Enther the subject number (in form of sub_x): ")

# Define maximum repetition
min_rep = 5
# Set the plot size
plt.rcParams['figure.figsize'] = [18, 12]

project_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project"
epoch_path = os.path.join(project_path, "EEG_Preprocessing", subject_name, subject_name + "-OC&CSD-AutoReject-epo.fif")
dic_path = os.path.join(project_path, "EventsID_Dictionary.csv")
statistics_path = os.path.join(project_path, "Stimuli")
# Read the Image dictionary
All_Images_df = pd.read_csv(dic_path, header=None)
All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))

# Get CE & SC list
with open(os.path.join(statistics_path, "lgn_statistics.pkl"), 'rb') as f:
    result = dill.load(f)
statistics_filename = result['filenames']
# The first dim: number of images, second: the RGB, third: number of random crops, fourth:  center vs. periphery
ce = result['CE']
ce_list = ce[:, :, 0, 0]
ce_list = ce_list.mean(axis=1)  # Average across RGB channels
sc = result['SC']
sc_list = sc[:, :, 0, 0]
sc_list = sc_list.mean(axis=1)

# Get Epochs
sub_epochs = mne.read_epochs(epoch_path)
sub_epochs = sub_epochs.pick_types(csd = True)
channel_names = sub_epochs.ch_names
# Get events filename
events = sub_epochs.events[:, 2]
events_filename = np.array([All_Images_dict[i] for i in events])
# oz_epochs = sub_epochs.pick_channels(ch_names=['Oz'])


# Get the subject unique images list
with open(os.path.join(statistics_path, "eeg_oads_stimulus_filenames.yml"), 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)
subjects = subjects[subject_name]
sub_sti = [i.split('\\')[1] for i in subjects if len(i) < 37]   # Stimuli images
sub_sti = np.array(sub_sti)
targets = [i.split('\\')[2] for i in subjects if len(i) >= 37]  # Target images
targets = np.array(targets)

# Get index of images in the events
Training_images_index = dict()
Testing_images_index = dict()
missing_images_counter = 0
for i in range(len(sub_sti)):
    cur_image_index = np.where(events_filename == sub_sti[i])[0]
    # cur_image_index = [a for a, b in enumerate(events_filename) if b == sub_sti[i]]
    if len(cur_image_index) > min_rep:
        Testing_images_index[events_filename[cur_image_index[0]]] = cur_image_index
    elif 0 < len(cur_image_index) <= min_rep:
        Training_images_index[events_filename[cur_image_index[0]]] = cur_image_index
    else:
        print("Didn't find ", sub_sti[i], " in the events\n")
        missing_images_counter += 1
if missing_images_counter == 0:
    print("NO missing images!\n")

# Use the index to make the design matrix
x = []
eeg_index = []
repetition_counter = 0
cur_image = 0
missing_statistics_counter = 0
Training_images_index_keys = list(Training_images_index.keys())
for i in range(len(events)):
    if repetition_counter >= min_rep:
        break
    cur_CE_para = [0] * min_rep  # Initiate CE parameters list for current image
    cur_SC_para = [0] * min_rep  # Initiate SC parameters list for current image
    while len(Training_images_index[Training_images_index_keys[cur_image]]) <= repetition_counter:
        if cur_image == len(Training_images_index_keys) - 1:
            cur_image = 0
            repetition_counter += 1
        else:
            cur_image += 1
        if repetition_counter >= min_rep:
            break
    if repetition_counter >= min_rep:
        break
    cur_image_name = Training_images_index_keys[cur_image]
    if cur_image_name not in statistics_filename:
        print(missing_statistics_counter, ": " "SC and CE for", cur_image_name, "are missing!\n")
        missing_statistics_counter += 1
    else:
        eeg_index.append(Training_images_index[cur_image_name][repetition_counter])
        statistics_index = statistics_filename.index(cur_image_name)  # Get the corresponding SC and CE index
        cur_CE_para[repetition_counter] = ce_list[statistics_index]
        cur_SC_para[repetition_counter] = sc_list[statistics_index]
        x.append(cur_CE_para + cur_SC_para)
    if cur_image == len(Training_images_index_keys) - 1:
        cur_image = 0
        repetition_counter += 1
    else:
        cur_image += 1
x = np.array(x)
if missing_statistics_counter == 0:
    print("NO missing statistics!\n")
# plt.pcolormesh(x)
# plt.imshow(x)
# plt.show()

# Fit regression model for each channel
r_sq_list = []
coef_list = []
for i in range(len(channel_names)):
    # Get the corresponding EEG data
    cur_channel = channel_names[i]
    channel_data = sub_epochs._data[eeg_index, i, :]
    cur_r_sq = []
    cur_coef = []
    # Fit linear regression model for each time points
    for j in range(channel_data.shape[1]):
        y = np.array(channel_data[:, j])
        regr = linear_model.LinearRegression().fit(x, y)
        # Save the Coefficients
        cur_coef.append(list(regr.coef_))
        cur_r_sq.append(regr.score(x, y))
    # plot the R^2 for current channel
    plt.figure(1)
    plt.plot(sub_epochs.times, cur_r_sq, label = cur_channel)
    r_sq_list. append(cur_r_sq)
    # plot CEs for current channel
    plt.figure(2)
    plt.plot(sub_epochs.times, np.array(cur_coef)[:, 0:min_rep].mean(axis = 1), label = cur_channel)
    # plot SCs for current channel
    plt.figure(3)
    plt.plot(sub_epochs.times, np.array(cur_coef)[:, min_rep:].mean(axis = 1), label = cur_channel)
    coef_list.append(cur_coef)
    del cur_r_sq, cur_coef
r_sq_list = np.array(r_sq_list)
coef_list = np.array(coef_list)

# plot the average
# mean_r_sq = r_sq_list.mean(axis = 0)
# plt.plot(sub_epochs.times, mean_r_sq, 'k', label = 'mean', linewidth=3)

# Setting parameters for the R^2 plot
plt.figure(1)
plt.xlabel('Time')
plt.ylabel('R^2')
plt.title(subject_name + "R^2")
plt.legend(fontsize="20")
plt.savefig(os.path.join(project_path, subject_name + "_R^2.png"))
# plt.show()
# Save r_sq
# np.save(os.path.join(project_path, subject_name + "_Rsq.npy"), r_sq_list)
# r_sq_df = pd.DataFrame(r_sq_list)
# r_sq_df.to_csv(os.path.join(project_path, subject_name + "_Rsq.csv"), header = False, index = False)

# Setting parameters for the CE and SC plot
plt.figure(2)
plt.xlabel('Time')
plt.ylabel('CE')
plt.title(subject_name + "β-coefficients (CE)")
plt.legend(fontsize="20")
plt.savefig(os.path.join(project_path, subject_name + "_CE.png"))
# plt.show()

plt.figure(3)
plt.xlabel('Time')
plt.ylabel('SC')
plt.title(subject_name + "β-coefficients (SC)")
plt.legend(fontsize="20")
plt.savefig(os.path.join(project_path, subject_name + "_SC.png"))
# plt.show()
# Save coefficients
# np.save(os.path.join(project_path, subject_name + "_βcoef.npy"), coef_list)


# Plot Oz
Oz_index = np.where(np.array(channel_names) == 'Oz')[0]
Oz_data = sub_epochs._data[eeg_index, Oz_index, :]
cur_r_sq = []
cur_coef = []
# Fit linear regression model for each time points
for j in range(Oz_data.shape[1]):
    y = np.array(Oz_data[:, j])
    regr = linear_model.LinearRegression().fit(x, y)
    # Save the Coefficients
    cur_coef.append(list(regr.coef_))
    cur_r_sq.append(regr.score(x, y))
cur_r_sq = np.array(cur_r_sq)
# plot the R^2
plt.figure(4)
plt.plot(sub_epochs.times, cur_r_sq)
plt.xlabel('Time')
plt.ylabel('R^2')
plt.title(subject_name + "R^2")
plt.savefig(os.path.join(project_path, subject_name + "_R^2(Oz).png"))
# Save Oz r_sq
# np.save(os.path.join(project_path, subject_name + "_Rsq(Oz).npy"), cur_r_sq)

# plot CEs for current channel
plt.figure(5)
plt.plot(sub_epochs.times, np.array(cur_coef)[:, 0:min_rep].mean(axis = 1), label = "CE")
# plot SCs for current channel
plt.plot(sub_epochs.times, np.array(cur_coef)[:, min_rep:].mean(axis = 1), label = "SC")
plt.xlabel('Time')
plt.ylabel('β-coefficients')
plt.title(subject_name + "β-coefficients")
plt.legend(fontsize="20")
plt.savefig(os.path.join(project_path, subject_name + "_β-coefficients(Oz).png"))
# Save coefficients
# np.save(os.path.join(project_path, subject_name + "_βcoef(Oz).npy"), cur_coef)