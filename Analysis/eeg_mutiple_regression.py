import dill
import numpy as np
import mne
import pandas as pd
import os
import yaml
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy.stats as stats

# Enter subject number
# subject_name = input("Enther the subject number (in form of sub_x): ")

# Define minimum repetition
min_rep = 5
# Set the plot size
plt.rcParams['figure.figsize'] = [18, 12]

project_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project"
dic_path = os.path.join(project_path, "EventsID_Dictionary.csv")
statistics_path = os.path.join(project_path, "Stimuli")
server_path = r"Z:\Projects\2023_Scholte_FMG1441"
# Get the subject unique images list
with open(os.path.join(statistics_path, "eeg_oads_stimulus_filenames.yml"), 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)
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

for i in range(21, 36):
    subject_name = "sub_" + str(i)
    # Get Epochs
    epoch_path = os.path.join(server_path, "Data", subject_name, "Preprocessed epochs", subject_name + "-OC&CSD-AutoReject-epo.fif")
    sub_epochs = mne.read_epochs(epoch_path)
    sub_epochs = sub_epochs.pick_types(csd = True)
    channel_names = sub_epochs.ch_names
    # Get events filename
    events = sub_epochs.events[:, 2]
    events_filename = np.array([All_Images_dict[i] for i in events])
    # oz_epochs = sub_epochs.pick_channels(ch_names=['Oz'])

    # Get the subject unique images list
    n_subjects = subjects[subject_name]
    sub_sti = [i.split('\\')[1] for i in n_subjects if len(i) < 37]  # Stimuli images
    sub_sti = np.array(sub_sti)
    targets = [i.split('\\')[2] for i in n_subjects if len(i) >= 37]  # Target images
    targets = np.array(targets)

    # Get index of images in the events
    Training_images_index = dict()
    Testing_images_index = dict()
    missing_images_counter = 0
    for j in range(len(sub_sti)):
        cur_image_index = np.where(events_filename == sub_sti[j])[0]    # Get the locations of current image
        # cur_image_index = [a for a, b in enumerate(events_filename) if b == sub_sti[i]]
        if len(cur_image_index) > min_rep:  # If it has presented more than 5 times, it goes to the test set
            Testing_images_index[events_filename[cur_image_index[0]]] = cur_image_index # Store every locations in the dictionary
        elif 0 < len(cur_image_index) <= min_rep:   # Otherwise, training set
            Training_images_index[events_filename[cur_image_index[0]]] = cur_image_index
        else:
            print("Didn't find ", sub_sti[j], " in the events\n")
            missing_images_counter += 1
        # del cur_image_index
    if missing_images_counter == 0:
        print("NO missing images from the events!\n")

    # Use the index to make the design matrix
    x = []    # Design matrix
    eeg_index = []
    repetition_counter = 0
    group_index = [0]
    cur_image = 0
    missing_statistics_counter = 0
    Training_images_index_keys = list(Training_images_index.keys())
    for j in range(len(events)):
        if repetition_counter >= min_rep:
            break
        cur_CE_para = [0] * min_rep  # Initiate CE parameters list for current image
        cur_SC_para = [0] * min_rep  # Initiate SC parameters list for current image
        while len(Training_images_index[Training_images_index_keys[cur_image]]) <= repetition_counter:
            if cur_image == len(Training_images_index_keys) - 1:
                group_index.append(j-missing_statistics_counter)
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
            eeg_index.append(Training_images_index[cur_image_name][repetition_counter]) # get the location of relevant event
            statistics_index = statistics_filename.index(cur_image_name)  # Get the corresponding SC and CE index
            cur_CE_para[repetition_counter] = ce_list[statistics_index]
            cur_SC_para[repetition_counter] = sc_list[statistics_index]
            x.append(cur_CE_para + cur_SC_para)
        if cur_image == len(Training_images_index_keys) - 1:
            group_index.append(j-missing_statistics_counter)
            cur_image = 0
            repetition_counter += 1
        else:
            cur_image += 1
        # del cur_CE_para, cur_SC_para
    if missing_statistics_counter == 0:
        print("NO missing statistics!\n")
    # Z-score each set of SC and CE in x
    x = np.array(x)
    # t = np.array(x)
    for j in range(repetition_counter):
        x[group_index[j]:group_index[j+1], j] = stats.zscore(x[group_index[j]:group_index[j+1], j])
        x[group_index[j]:group_index[j+1], j + min_rep] = stats.zscore(x[group_index[j]:group_index[j+1], j + min_rep])
    # plt.imshow(x)
    # plt.show()

    # Fit regression model for each channel
    r_sq_list = []
    coef_list = []
    for j in range(len(channel_names)):
        # Get the corresponding EEG data
        cur_channel = channel_names[j]
        channel_data = sub_epochs._data[eeg_index, j, :]
        cur_r_sq = []
        cur_coef = []
        # Fit linear regression model for each time points
        for k in range(channel_data.shape[1]):
            y = stats.zscore(np.array(channel_data[:, k]))
            regr = linear_model.LinearRegression().fit(x, y)
            # Save the Coefficients
            cur_coef.append(list(regr.coef_))
            cur_r_sq.append(regr.score(x, y))
        # plot the R^2 for current channel
        # plt.figure(1)
        # plt.plot(sub_epochs.times, cur_r_sq, label = cur_channel)
        r_sq_list. append(cur_r_sq)
        # plot CEs for current channel
        # plt.figure(2)
        # plt.plot(sub_epochs.times, np.array(cur_coef)[:, 0:min_rep].mean(axis = 1), label = cur_channel)
        # plot SCs for current channel
        # plt.figure(3)
        # plt.plot(sub_epochs.times, np.array(cur_coef)[:, min_rep:].mean(axis = 1), label = cur_channel)
        coef_list.append(cur_coef)
        # del cur_r_sq, cur_coef
    r_sq_list = np.array(r_sq_list)
    coef_list = np.array(coef_list)

    # plot the average
    # mean_r_sq = r_sq_list.mean(axis = 0)
    # plt.plot(sub_epochs.times, mean_r_sq, 'k', label = 'mean', linewidth=3)

    # Setting parameters for the R^2 plot
    plt.figure(1).clear()
    plt.figure(1)
    plt.plot(sub_epochs.times, r_sq_list.T)
    plt.xlabel('Time', fontsize = 25)
    plt.ylabel('R^2', fontsize = 25)
    plt.title(subject_name + "R^2", fontsize = 30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(fontsize="20")
    plt.savefig(os.path.join(project_path, subject_name + "_R^2.png"))
    # plt.show()
    # Save r_sq
    np.save(os.path.join(project_path, subject_name + "_Rsq.npy"), r_sq_list)

    # Setting parameters for the CE and SC plot
    plt.figure(2).clear()
    plt.figure(2)
    plt.plot(sub_epochs.times, coef_list[:,:, 0:min_rep].mean(axis = 2).T)
    plt.xlabel('Time', fontsize = 25)
    plt.ylabel('β-coefficients (CE)', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(subject_name + "β-coefficients (CE)", fontsize = 30)
    plt.legend(fontsize="20")
    plt.savefig(os.path.join(project_path, subject_name + "_CE.png"))
    # plt.show()

    plt.figure(3).clear()
    plt.figure(3)
    plt.plot(sub_epochs.times, coef_list[:, :, min_rep:].mean(axis = 2).T)
    plt.xlabel('Time', fontsize = 25)
    plt.ylabel('β-coefficients (SC)', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(subject_name + "β-coefficients (SC)", fontsize = 30)
    plt.legend(fontsize="20")
    plt.savefig(os.path.join(project_path, subject_name + "_SC.png"))
    # plt.show()
    # Save coefficients
    np.save(os.path.join(project_path, subject_name + "_βcoef.npy"), coef_list)


    # Plot Oz
    Oz_index = np.where(np.array(channel_names) == 'Oz')[0]
    Oz_data = sub_epochs._data[eeg_index, Oz_index, :]
    cur_r_sq = []
    cur_coef = []
    # Fit linear regression model for each time points
    for j in range(Oz_data.shape[1]):
        y = stats.zscore(np.array(Oz_data[:, j]))
        regr = linear_model.LinearRegression().fit(x, y)
        # Save the Coefficients
        cur_coef.append(list(regr.coef_))
        cur_r_sq.append(regr.score(x, y))
    cur_r_sq = np.array(cur_r_sq)
    # plot the Oz R^2
    plt.figure(4).clear()
    plt.figure(4)
    plt.plot(sub_epochs.times, cur_r_sq)
    plt.xlabel('Time', fontsize = 25)
    plt.ylabel('R^2', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(subject_name + "_R^2 (Oz)", fontsize = 30)
    plt.savefig(os.path.join(project_path, subject_name + "_R^2(Oz).png"))
    # Save Oz r_sq
    np.save(os.path.join(project_path, subject_name + "_Rsq(Oz).npy"), cur_r_sq)

    # plot CEs for channel Oz
    plt.figure(5).clear()
    plt.figure(5)
    plt.plot(sub_epochs.times, np.array(cur_coef)[:, 0:min_rep].mean(axis = 1), label = "CE")
    # plot SCs for current channel
    plt.plot(sub_epochs.times, np.array(cur_coef)[:, min_rep:].mean(axis = 1), label = "SC")
    plt.xlabel('Time', fontsize = 25)
    plt.ylabel('β-coefficients', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(subject_name + "β-coefficients (Oz)", fontsize = 30)
    plt.legend(fontsize="20")
    plt.savefig(os.path.join(project_path, subject_name + "_β-coefficients(Oz).png"))
    # Save coefficients
    np.save(os.path.join(project_path, subject_name + "_βcoef(Oz).npy"), cur_coef)

    # Plot the design matrix
    plt.figure(6).clear()
    plt.figure(6)
    # Add intercepts column
    Intercept_constant = np.ones((x.shape[0],1))
    x = np.hstack((Intercept_constant,x))
    # # Save Design matrix
    np.save(os.path.join(project_path, subject_name + "_Design_Matrix.npy"), x)
    new_x = np.ma.masked_where(x == 0, x)
    cmap = plt.get_cmap('inferno')
    cmap.set_bad(color = 'white')
    plt.pcolormesh(new_x, cmap = cmap)
    plt.xlabel('Parameters', fontsize = 25)
    plt.ylabel('Training images', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(subject_name + "Design Matrix", fontsize = 30)
    plt.colorbar()
    # plt.legend(fontsize="20")
    plt.savefig(os.path.join(project_path, subject_name + "_Design Matrix.png"))

    # #Delete the var
    # del epoch_path, sub_epochs, events, events_filename, n_subjects, sub_sti, targets, Training_images_index, Testing_images_index, \
    #     x, eeg_index, Training_images_index_keys, group_index, r_sq_list, coef_list, cur_r_sq, cur_coef