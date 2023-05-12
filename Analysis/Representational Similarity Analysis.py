import torch
import numpy as np
import mne
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt


# Enter subject number
subject_name = input("Enther the subject number (in form of sub_x): ")

# Define minimum repetition
min_rep = 5
# Set the plot size
plt.rcParams['figure.figsize'] = [24, 16]

project_path = r"C:\Users\15202\OneDrive\C_\University of Amsterdam\Intern\CAVA_project"
server_path = r"Y:\Projects\2023_Scholte_FMG1441"
dic_path = os.path.join(project_path, "EventsID_Dictionary.csv")
statistics_path = os.path.join(project_path, "Stimuli")
epoch_path = os.path.join(server_path, "Data", subject_name, "Preprocessed epochs", subject_name + "-OC&CSD-AutoReject-epo.fif")
Model_path = os.path.join(project_path, "Res-net", "resnet50", "rgb", "best_model_23-03-23-17-14-41.pth")

# Get Epochs
sub_epochs = mne.read_epochs(epoch_path)
sub_epochs = sub_epochs.pick_types(csd = True)
channel_names = sub_epochs.ch_names
# Get events filename
events = sub_epochs.events[:, 2]
All_Images_df = pd.read_csv(dic_path, header=None)      # Read the Image dictionary
All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))     # Read the Image dictionary
events_filename = np.array([All_Images_dict[i] for i in events])
oz_epochs = sub_epochs.pick_channels(ch_names=['Oz'])

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
    print("NO missing images from the events!\n")
Training_images_index_list = list(Training_images_index.values())
Testing_images_index_list = list(Testing_images_index.values())

# Make the representational dissimilarity matrix (RDM)
RDM = []    # Initialize the matrix
for i in range(len(Training_images_index_list)):
    cur_dis_list = []
    row_data = oz_epochs._data[Training_images_index_list[i], 0, :].mean(axis = 0)
    for j in range(len(Training_images_index_list)):
        if i > j:
            cur_dis_list.append(RDM[j][i])
            continue
        column_data = oz_epochs._data[Training_images_index_list[j], 0, :].mean(axis = 0)
        cur_coef = np.corrcoef(row_data, column_data)[0, 1]
        cur_dis_list.append(1-cur_coef)
    RDM.append(cur_dis_list)
RDM = np.array(RDM)

# Visualize RDM
plt.figure(1)
plt.pcolormesh(RDM)
plt.xlabel('Training Images', fontsize = 25)
plt.ylabel('Training images', fontsize = 25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(subject_name + "_Representational Dissimilarity Matrix", fontsize = 30)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=25)
plt.savefig(os.path.join(project_path, subject_name + "_Representational Dissimilarity Matrix.png"))

# Load Res-net model
Resnet_model = torch.load(Model_path, map_location=torch.device('cpu'))
Resnet_model.eval()
