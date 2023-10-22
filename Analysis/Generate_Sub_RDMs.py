import numpy as np
import mne
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import copy
import scipy.stats
import multiprocessing
import time
import tqdm
import pickle


import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pytorch_utils.pytorch_utils import ToJpeg, ToOpponentChannel, collate_fn, record_activations, evaluate
from oads_access.oads_access import OADS_Access, OADSImageDataset


# Define relevant paths
project_path = r"/home/c14271389"
server_path = r"/home/c14271389/FMG-folder"
dic_path = os.path.join(project_path, "EventsID_Dictionary.csv")
statistics_path = os.path.join(project_path, "Stimuli")


# Define repetition minimum
min_rep = 5
num_workers = 12
use_crops = False   #False
# Load images file names
with open(os.path.join(statistics_path, "eeg_oads_stimulus_filenames.yml"), 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)
# Load OADS dataset
oads = OADS_Access(basedir=f'/home/Public/Datasets/oads', n_processes=num_workers)
train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=use_crops)


# Generate subject RDMs
def generate_sub_rdm(argx):
    subject_name = "sub_" + str(argx)

    # Get Epochs
    epoch_path = os.path.join(server_path, subject_name, "Preprocessed epochs", subject_name + "-OC&CSD-AutoReject-epo.fif")
    sub_epochs = mne.read_epochs(epoch_path)
    sub_epochs = sub_epochs.pick_types(csd = True)

    # Pick Posterior channels
    channel_names = sub_epochs.ch_names
    posterior_lobe = ['P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'F5', 'F6']
    # posterior_channel = [i for i,j in enumerate(channel_names) if j in posterior_lobe]
    posterior_epochs = sub_epochs.pick_channels(ch_names=posterior_lobe)

    # Get events filename
    events = posterior_epochs.events[:, 2]
    All_Images_df = pd.read_csv(dic_path, header=None)      # Read the Image dictionary
    All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))     # Read the Image dictionary
    events_filename = np.array([All_Images_dict[i][:-5] for i in events])
    
    # Get images names
    cur_sub = copy.deepcopy(subjects[subject_name])
    for i in range(len(cur_sub)):
        if len(cur_sub[i]) < 37:
            cur_sub[i] = cur_sub[i].split('\\')[1][:-5]
        else:
            cur_sub[i] = cur_sub[i].split('\\')[2][:-5]

    # Get index of images from the events and model
    Images_index = dict()
    missing_images_counter = 0
    for i in range(len(cur_sub)):
        cur_image_index = np.where(events_filename == cur_sub[i])[0]
        # cur_image_index = [a for a, b in enumerate(events_filename) if b == sub_sti[i]]
        if (len(cur_image_index) > 0) and (cur_sub[i] in (train_ids + test_ids + val_ids)):
            Images_index[cur_sub[i]] = cur_image_index
        else:
            # print("Didn't find ", cur_sub[i], " in the events\n")
            missing_images_counter += 1
    Image_names = list(Images_index.keys())
    Images_index_list = list(Images_index.values())

    # Make the RDM for subjects on each timepoints
    sub_RDM = []
    for timepoints in range(posterior_epochs._data.shape[2]):
        cur_RDM = []    # Initialize the matrix
        for i in range(len(Image_names)):
            cur_dis_list = []
            row_data = posterior_epochs._data[Images_index[Image_names[i]], :, timepoints].mean(axis = 0)
            for j in range(len(Image_names)):
                if i > j:
                    cur_dis_list.append(cur_RDM[j][i])
                    continue
                elif i == j:
                    cur_dis_list.append(0.0)
                    continue
                else:
                    column_data = posterior_epochs._data[Images_index[Image_names[j]], :, timepoints].mean(axis = 0)
                    cur_coef = 1 - scipy.stats.pearsonr(row_data, column_data)[0]
                    cur_dis_list.append(cur_coef)
            cur_RDM.append(cur_dis_list)
        sub_RDM.append(cur_RDM)
        print(f"Done with timepoint#{timepoints}")

    np.save(os.path.join(project_path, "Sub_RDMs(post)", subject_name + "_RDM(post).npy"), np.array(sub_RDM))


# Multiprocessing
start_sub = 5
end_sub = 35
with multiprocessing.Pool(end_sub - start_sub + 1) as p:
    start = time.time()
    sub_iter = list(range(start_sub, end_sub+1))
    p.map(generate_sub_rdm, iterable = sub_iter)
    end = time.time()
    print(end-start)