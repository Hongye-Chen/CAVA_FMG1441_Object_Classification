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


# Generate model RDMs for each subjecet
def generate_model_rdm(argx):
    subject_name = "sub_" + str(argx)
    activation_path = os.path.join(project_path, "Model_activations(ImageNet)", "Raw_img", subject_name + "_activations(RAW_ImageNet_lastlayer).pkl")

    # Read dictionary pkl file
    with open(activation_path, 'rb') as fp:
        activations = pickle.load(fp)
    model_act = activations
    # Build the RDM
    model_RDM = [] # Initialization
    for i in range(len(model_act)):
        cur_dis_list = []
        row_data = model_act[i]
        for j in range(len(model_act)):
            if i > j:
                cur_dis_list.append(model_RDM[j][i])
                continue
            elif i == j:
                cur_dis_list.append(0.0)
                continue
            else:
                column_data = model_act[j]
                # print(subject_name, j)
                cur_coef = 1 - scipy.stats.pearsonr(row_data, column_data)[0]
                cur_dis_list.append(cur_coef)
        model_RDM.append(cur_dis_list)
    # Save the model RDMs
    model_RDM = np.array(model_RDM)
    print(f"{subject_name}: {model_RDM.shape}")
    np.save(os.path.join(project_path, "Model_RDMs(ImageNet)", "Raw_Image", subject_name + "_model(RAW_ImageNet)_RDM(last_layer).npy"), model_RDM)
    # Clean out spaces
    del activations


# Multiprocessing
start_sub = 5
end_sub = 35
with multiprocessing.Pool(end_sub - start_sub + 1) as p:
    start = time.time()
    sub_iter = list(range(start_sub, end_sub+1))
    p.map(generate_model_rdm, iterable = sub_iter)
    end = time.time()
    print(end-start)