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
from scipy import stats

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pytorch_utils.pytorch_utils import ToJpeg, ToOpponentChannel, collate_fn, record_activations, evaluate
from oads_access.oads_access import OADS_Access, OADSImageDataset

# Set the paths
project_path = ""
server_path = ""
dic_path = os.path.join(project_path, "EventsID_Dictionary.csv")
statistics_path = os.path.join(project_path, "Stimuli")
# epoch_path = os.path.join(server_path, subject_name, "Preprocessed epochs", subject_name + "-OC&CSD-AutoReject-epo.fif")
# Model_path = os.path.join(project_path, "Res-net", "resnet50", "rgb", "best_model_23-03-23-17-14-41.pth")

oads = OADS_Access(basedir=f'', n_processes=12)
train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=False)

with open(os.path.join(statistics_path, "eeg_oads_stimulus_filenames.yml"), 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)

shared_images = np.intersect1d(np.array(subjects["sub_20"]), np.array(subjects["sub_35"]))

start_sub = 20
end_sub = 35
with multiprocessing.Pool(end_sub - start_sub + 1) as p:
    start = time.time()
    sub_iter = list(range(start_sub, end_sub+1))
    p.map(shared, iterable = sub_iter)
    end = time.time()
    print(end-start)

shared_rdms = []
for i in range(20, 36):
    sub = "sub_" + str(i)
    cur_sub = np.load(os.path.join(project_path, "Noise_Ceiling", sub + "_RDM(shared).npy"))
    shared_rdms.append(cur_sub)
shared_rdms = np.array(shared_rdms)
print(shared_rdms.shape)

all_low = []
all_high = []
for timepoints in range(513):
    cur_time = shared_rdms[:, timepoints, :, :]
    zscore = np.array([scipy.stats.zscore(i, axis = None) for i in cur_time])
    avg_cur = zscore.mean(axis = 0)
    lower_bound = []
    upper_bound = []
    for i in range(20, 36):
        cur_sub = shared_rdms[i-20, timepoints, :, :]
        cur_sub = scipy.stats.zscore(cur_sub, axis = None)
        left_out = list(range(0,i-20)) + list(range(i-19,16))
        left_out = shared_rdms[left_out, timepoints, :, :]
        z_left_out = np.array([scipy.stats.zscore(i, axis = None) for i in left_out])
        avg_left_out = z_left_out.mean(axis = 0)

        high_cur = 1 - scipy.stats.pearsonr(avg_cur.flatten(), cur_sub.flatten())[0]
        upper_bound.append(high_cur)
        low_cur = 1 - scipy.stats.pearsonr(avg_left_out.flatten(), cur_sub.flatten())[0]
        lower_bound.append(low_cur)
    all_high.append(np.array(upper_bound).mean())
    all_low.append(np.array(lower_bound).mean())

np.save(os.path.join(project_path, "upper_bound.npy"), np.array(all_high))
np.save(os.path.join(project_path, "lower_bound.npy"), np.array(all_low))


def shared(argx):
    subject_name = "sub_" + str(argx)
    epoch_path = os.path.join(server_path, subject_name, "Preprocessed epochs", subject_name + "-OC&CSD-AutoReject-epo.fif")
    # Get Epochs
    sub_epochs = mne.read_epochs(epoch_path)
    sub_epochs = sub_epochs.pick_types(csd = True)
    channel_names = sub_epochs.ch_names
    posterior_lobe = ['P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'F5', 'F6']
    # posterior_channel = [i for i,j in enumerate(channel_names) if j in posterior_lobe]
    posterior_epochs = sub_epochs.pick_channels(ch_names=posterior_lobe)
    # Get events filename
    events = posterior_epochs.events[:, 2]
    All_Images_df = pd.read_csv(dic_path, header=None)      # Read the Image dictionary
    All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))     # Read the Image dictionary
    events_filename = np.array([All_Images_dict[i][:-5] for i in events])
    # oz_epochs = sub_epochs.pick_channels(ch_names=['Oz'])

    # Get the subject images list
    cur_sub = copy.deepcopy(shared_images)
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
    # if missing_images_counter == 0:
    #     print("NO missing images from the events!\n")
    # else:
    #     print(f"There are {missing_images_counter} missing images from the events.")
    Image_names = list(Images_index.keys())
    Images_index_list = list(Images_index.values())

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
        # print(f"Done with timepoint#{timepoints}")
    print(f"Shape of subject {subject_name} is: {len(Image_names)}")

    np.save(os.path.join(project_path, "Noise_Ceiling", subject_name + "_RDM(shared).npy"), np.array(sub_RDM))

