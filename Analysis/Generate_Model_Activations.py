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
from PIL import Image


import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pytorch_utils.pytorch_utils import ToJpeg, ToOpponentChannel, collate_fn, record_activations, evaluate
from oads_access.oads_access import OADS_Access, OADSImageDataset

class EdgeResize(object):
    def __init__(self, size, as_tensor: bool = False):
        super().__init__()

        if type(size) is int:
            self.size = (size, size)
        elif type(size) is tuple or type(size) is list:
            self.size = size
        self.resize = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(self.size)])

        self.as_tensor = as_tensor

    def __call__(self, sample):
        arr = []
        # print(type(sample))
        for _x in sample.transpose((2, 0, 1)):
            res = np.array(self.resize(Image.fromarray(_x)))
            arr.append(res)

        arr = np.dstack(arr)

        if self.as_tensor:
            return torch.tensor(arr)

        return arr

project_path = r"/home/c14271389"
server_path = r"/home/c14271389/FMG-folder"
dic_path = os.path.join(project_path, "EventsID_Dictionary.csv")
statistics_path = os.path.join(project_path, "Stimuli")

min_rep = 5
# Load dataset / dataloader
num_workers = 24
oads = OADS_Access(basedir=f'/home/Public/Datasets/oads', n_processes=num_workers)
size = (2155, 1440)      #Smaller
use_crops = False   #False
model_type = 'resnet50' # resnet18
image_representation = 'rgb' # coc
use_rgbedges = False
identifier = {'resnet50': {'rgb': {'raw': '2023-03-23'}}}
image_quality = 'raw'

model_path = os.path.join(f'{project_path}', model_type, image_representation, identifier[model_type][image_representation][image_quality])
model_path = os.path.join(model_path, [x for x in os.listdir(model_path) if x.startswith('best_model')][0])
model_path = "/home/c14271389/resnet50/resnet50_imagenet.pth"
# model_path = os.path.join(home_path,"resnet50", "coc", "best_model_23-03-23-17_33_58.pth")
gpu_name = 'cuda:1'
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')
torch.cuda.empty_cache()
batch_size = 4 #512 # 512
output_channels = 1000 # 21 if it's COC/EdgeMap model, 19 if RGB



if model_type == 'resnet50':
    # Init model
    model = resnet50()

    # Init number of input nodes, here 3 =RGB or COC
    model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    
    # Init number of output nodes, here 19
    model.fc = torch.nn.Linear(in_features=2048, out_features=output_channels, bias=True)
    
    # Load state dict (pretrained model)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.module
elif model_type == 'resnet18':
    model = resnet18()
    model.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=model.conv1.out_channels, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    model.fc = torch.nn.Linear(
        in_features=512, out_features=output_channels, bias=True)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.module

# Move model to GPU if available
model = model.to(device)

return_nodes = {
    # node_name: user-specified key for output dict
    'layer1.2.relu_2': 'layer1',
    'layer2.3.relu_2': 'layer2',
    'layer3.5.relu_2': 'layer3',
    'layer4.2.relu_2': 'layer4',
    'flatten': 'feature',
    # 'fc': 'classification'
}
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)


output_channels = len(oads.get_class_mapping())
class_index_mapping = {}
index_label_mapping = {}
for index, (key, item) in enumerate(list(oads.get_class_mapping().items())):
    class_index_mapping[key] = index
    index_label_mapping[index] = item


# OADS Crops (400,400) mean, std (RGB)
mean = [0.3410, 0.3123, 0.2787]
std = [0.2362, 0.2252, 0.2162]
# OADS Crops (400,400) mean, std (COC)
# mean = [0.30080804, 0.02202087, 0.01321364]
# std = [0.06359817, 0.01878176, 0.0180428]
# Edge map mean, std


# Get the custom dataset and dataloader
print(f"Getting data loaders")
transform_list = []
if use_rgbedges:
    transform_list.append(EdgeResize(size))
else:
    transform_list.append(transforms.Resize(size))

# Apply color opponnent channel representation
if image_representation == 'coc':
    transform_list.append(ToOpponentChannel())

transform_list.append(transforms.ToTensor())
transform_list.append(transforms.Normalize(mean, std))  # Skip if using EdgeMap

transform = transforms.Compose(transform_list)


train_ids, val_ids, test_ids = oads.get_train_val_test_split_indices(use_crops=use_crops)


with open(os.path.join(statistics_path, "eeg_oads_stimulus_filenames.yml"), 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)


# Generate sub activations
for sub in range(5,36):
    subject_name = "sub_" + str(sub)
    epoch_path = os.path.join(server_path, subject_name, "Preprocessed epochs", subject_name + "-OC&CSD-AutoReject-epo.fif")
    # Get Epochs
    sub_epochs = mne.read_epochs(epoch_path)
    sub_epochs = sub_epochs.pick_types(csd = True)
    channel_names = sub_epochs.ch_names
    # Get events filename
    events = sub_epochs.events[:, 2]
    All_Images_df = pd.read_csv(dic_path, header=None)      # Read the Image dictionary
    All_Images_dict = dict(zip(All_Images_df[1], All_Images_df[0]))     # Read the Image dictionary
    events_filename = np.array([All_Images_dict[i][:-5] for i in events])
    # oz_epochs = sub_epochs.pick_channels(ch_names=['Oz'])

    # Get the subject images list
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
    # if missing_images_counter == 0:
    #     print("NO missing images from the events!\n")
    # else:
    #     print(f"There are {missing_images_counter} missing images from the events.")
    Image_names = list(Images_index.keys())
    Images_index_list = list(Images_index.values())


    # Build the Representational Dissimilarity Matrix (RDM) for model
    # Created custom OADS datasets
    traindataset = OADSImageDataset(oads_access=oads, item_ids=Image_names, use_crops=use_crops, preload_all=False, target=None,
                                    class_index_mapping=class_index_mapping, transform=transform, device=device, return_index = True)
    # Create loaders - shuffle training set, but not validation or test set
    trainloader = DataLoader(traindataset, collate_fn=collate_fn,
                                batch_size=4, shuffle=False, num_workers=oads.n_processes)  #shuffle = Flase
    # Extract Activations
    activations = record_activations(loader=trainloader, models=[(model_type, feature_extractor)], device=device, n_nodes=20, layer_names=return_nodes.values(), extract_pixel_layer=True)
    # Save activations
    activation_path = os.path.join(project_path, "Model_activations(ImageNet)", "Raw_img", subject_name + "_activations(RAW_ImageNet_lastlayer).pkl")
    with open(activation_path, 'wb') as fp:
        pickle.dump(activations['resnet50_feature'], fp)
        print(f'{subject_name} dictionary saved successfully to file')

    