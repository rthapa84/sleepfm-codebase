import torch
import cv2
import os
import numpy as np
import pandas
import pickle
import torchvision
import random
import math
import time
from config import (CHANNEL_DATA, ALL_CHANNELS, CHANNEL_DATA_IDS)
from loguru import logger
import shutil
import sys
sys.path.append("../")
from config import EVENT_TO_ID, LABELS_DICT


class EventDatasetSupervised(torchvision.datasets.VisionDataset):
    def __init__(self, root, split="train", modality_type="sleep_stages"):
        start = time.time()
        self.split = split
        self.modality_type = modality_type

        with open(root, "rb") as f:
            self.dataset = pickle.load(f)

        if split == "combined":
            self.dataset = self.dataset["pretrain"] + self.dataset["train"]
        else:
            self.dataset = self.dataset[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data_path = self.dataset[index][0]
        event = self.dataset[index][1]
        event_num = LABELS_DICT[event]
        
        data = np.load(data_path)

        if self.modality_type == "respiratory":
            data = data[CHANNEL_DATA_IDS["Respiratory"]]
        elif self.modality_type == "sleep_stages":
            data = data[CHANNEL_DATA_IDS["Sleep_Stages"]]
        elif self.modality_type == "ekg":
            data = data[CHANNEL_DATA_IDS["EKG"]]
        elif self.modality_type == "combined":
            all_ids = CHANNEL_DATA_IDS["Respiratory"] + CHANNEL_DATA_IDS["Sleep_Stages"] + CHANNEL_DATA_IDS["EKG"]
            data = data[all_ids]
        else:
            raise ValueError(f'Modality type "{self.modality_type}" is not recognized.')
        
        return data, event_num


class EventDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, split="train", modality_type=["respiratory", "sleep_stages", "ekg"]):
        start = time.time()
        self.split = split
        if isinstance(modality_type, list):
            self.modality_type = modality_type
        else:
            self.modality_type = [modality_type]

        with open(root, "rb") as f:
            self.dataset = pickle.load(f)

        self.dataset = self.dataset[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data_path = self.dataset[index][0]
        data = np.load(data_path)
        
        target: Any = []
        for t in self.modality_type:
            if t == "respiratory":
                resp_data = data[CHANNEL_DATA_IDS["Respiratory"]]
                target.append(resp_data)
            elif t == "sleep_stages":
                sleep_data = data[CHANNEL_DATA_IDS["Sleep_Stages"]]
                target.append(sleep_data)
            elif t == "ekg":
                ekg_data = data[CHANNEL_DATA_IDS["EKG"]]
                target.append(ekg_data)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')
        
        return target


_cache = {}
def cache_csv(path, sep=None):
    if path in _cache:
        return _cache[path]
    else:
        x = pandas.read_csv(path, sep=sep)
        _cache[path] = x
        return x

_cache = {}
def cache_pkl(path):
    if path in _cache:
        return _cache[path]
    else:
        with open(path, "rb") as f:
            x = pickle.load(f)
        _cache[path] = x
        return x