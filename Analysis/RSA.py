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
import pickle
import tqdm

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
# Model_path = os.path.join(project_path, "Res-net", "resnet50", "rgb", "best_model_23-03-23-17-14-41.pth")


def rsa(argx):
    subject = "sub_" + str(argx)
    # Load the model RDMs and the subjecet RDM
    model_RDM = np.load(os.path.join(project_path, "Model_RDMs(ImageNet)", "Raw_Image", subject + "_model(RAW_ImageNet)_RDM(last_layer).npy"))
    # model_RDM = np.load(os.path.join(project_path, "Model_RDMs(Niklas)", subject + "_model(Niklas)_RDM(last_layer).npy"))
    sub_RDM = np.load(os.path.join(project_path, "Sub_RDMs(post)", subject + "_RDM(post).npy"))
    shuffled_RSA_matrix = []
    RSA_list = []
    RSA_p = []

    upper_indices = np.triu_indices(sub_RDM.shape[1]) # Select the upper above of the matrix

    for timepoints in range(sub_RDM.shape[0]):
        cur_sub_RDM = sub_RDM[timepoints]
        cur_RSA = 1 - scipy.stats.spearmanr(cur_sub_RDM[upper_indices], model_RDM[upper_indices])[0]
        RSA_list.append(cur_RSA)
        # shuffle the sub_RDM 1000 times and recompute the correlation
        shuffled_RSA = []
        reorder = np.array(range(sub_RDM.shape[1]))
        for shuffle in range(1000):
            np.random.shuffle(reorder)
            cur_sub_RDM[:, :] = cur_sub_RDM[reorder, :]
            cur_sub_RDM[:, :] = cur_sub_RDM[:, reorder]
            # recalculate the RSA
            new_RSA = 1 - scipy.stats.spearmanr(cur_sub_RDM[upper_indices], model_RDM[upper_indices])[0]
            shuffled_RSA.append(new_RSA)
        shuffled_RSA_matrix.append(shuffled_RSA)
        shuffled_RSA = np.array(shuffled_RSA)
        # compute and record the p-value
        cur_p = scipy.stats.norm.cdf(cur_RSA, shuffled_RSA.mean(), shuffled_RSA.std())
        if cur_RSA > shuffled_RSA.mean():
            cur_p = 1 - cur_p
        RSA_p.append(cur_p)
        print(f"Done with {subject}: timepoint#{timepoints}, correlation: {cur_RSA}, P-value: {cur_p}")

    # Save the RSA results
    RSA_array = np.array(RSA_list)
    shuffled_RSA_matrix = np.array(shuffled_RSA_matrix)
    RSA_p = np.array(RSA_p)
    np.save(os.path.join(project_path, "Subject_RSA(ImageNet-Posterior)", "Raw_Image", subject, subject + "_RSA(RAW_ImageNet-Posterior).npy"), RSA_array)
    np.save(os.path.join(project_path, "Subject_RSA(ImageNet-Posterior)", "Raw_Image", subject, subject + "_shuffled_RSA_matrix(RAW_ImageNet-Posterior).npy"), shuffled_RSA_matrix)
    np.save(os.path.join(project_path, "Subject_RSA(ImageNet-Posterior)", "Raw_Image", subject, subject + "_RSA_Pvalue(RAW_ImageNet-Posterior).npy"), RSA_p)


# Multiprocessing
start_sub = 5
end_sub = 35
with multiprocessing.Pool(end_sub - start_sub + 1) as p:
    start = time.time()
    sub_iter = list(range(start_sub, end_sub+1))
    p.map(rsa, iterable = sub_iter)
    end = time.time()
    print(end-start)