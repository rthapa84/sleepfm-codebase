import time
import torch
import torchvision
import os
import click
import tqdm
import math
import shutil
import datetime
import numpy as np
from loguru import logger
import pickle
import math

import sys
sys.path.append("../model")
import models
from config import (CONFIG, CHANNEL_DATA, 
                    ALL_CHANNELS, CHANNEL_DATA_IDS, 
                    PATH_TO_PROCESSED_DATA)

from dataset import EventDataset as Dataset 

@click.command("generate_eval_embed")
@click.argument("output_file", type=click.Path())
@click.option("--dataset_dir", type=str, default=None)
@click.option("--dataset_file", type=str, default="dataset_events_-1.pickle")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=2)
@click.option("--splits", type=click.STRING, default=['train', 'valid', 'test'], help='Specify the data splits (train, valid, test).')
def generate_eval_embed(
    output_file,
    dataset_dir,
    dataset_file,
    batch_size,
    num_workers,
    splits
):
    if dataset_dir == None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    output_dir = os.path.join(dataset_dir, f"{output_file}")

    device = torch.device("cuda")
    splits = splits.split(",")

    path_to_data = dataset_dir

    dataset = {
        split: Dataset(os.path.join(path_to_data, dataset_file), split=split, modality_type=["respiratory", "sleep_stages", "ekg"])
        for split in splits
    }

    in_channel = len(CHANNEL_DATA_IDS["Respiratory"])
    model_resp = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
    model_resp.fc = torch.nn.Linear(model_resp.fc.in_features, 512)
    if device.type == "cuda":
        model_resp = torch.nn.DataParallel(model_resp)
    model_resp.to(device)

    in_channel = len(CHANNEL_DATA_IDS["Sleep_Stages"])
    model_sleep = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
    model_sleep.fc = torch.nn.Linear(model_sleep.fc.in_features, 512)
    if device.type == "cuda":
        model_sleep = torch.nn.DataParallel(model_sleep)
    model_sleep.to(device)

    in_channel = len(CHANNEL_DATA_IDS["EKG"])
    model_ekg = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
    model_ekg.fc = torch.nn.Linear(model_ekg.fc.in_features, 512)
    if device.type == "cuda":
        model_ekg = torch.nn.DataParallel(model_ekg)
    model_ekg.to(device)

    checkpoint = torch.load(os.path.join(output_dir, "best.pt"))
    temperature = checkpoint["temperature"]

    model_resp.load_state_dict(checkpoint["respiratory_state_dict"])
    model_resp.eval()

    model_sleep.load_state_dict(checkpoint["sleep_stages_state_dict"])
    model_sleep.eval()

    model_ekg.load_state_dict(checkpoint["ekg_state_dict"])
    model_ekg.eval()

    path_to_save = os.path.join(output_dir, "eval_data")
    os.makedirs(path_to_save, exist_ok=True)

    for split in splits:
        dataloader = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
        save_interval = math.ceil(len(dataloader) / 8)
        counter = 0
        emb = [[], [], []]
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader), desc=("Embeddings for " + split)) as pbar:
                for (i, (resp, sleep, ekg)) in enumerate(dataloader):
                    resp = resp.to(device, dtype=torch.float)
                    sleep = sleep.to(device, dtype=torch.float)
                    ekg = ekg.to(device, dtype=torch.float)

                    emb[0].append(torch.nn.functional.normalize(model_resp(resp)).detach().cpu())
                    emb[1].append(torch.nn.functional.normalize(model_sleep(sleep)).detach().cpu())
                    emb[2].append(torch.nn.functional.normalize(model_ekg(ekg)).detach().cpu())

                    pbar.update()
        
        emb = list(map(torch.concat, emb))
        dataset_prefix = dataset_file.split(".")[0]
        with open(os.path.join(path_to_save, f"{dataset_prefix}_{split}_emb.pickle"), 'wb') as file:
            pickle.dump(emb, file)



if __name__ == '__main__':
    generate_eval_embed()



