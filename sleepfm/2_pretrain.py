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
import numpy as np

import sys
sys.path.append("../model")
import models
from config import (CONFIG, CHANNEL_DATA, 
                    ALL_CHANNELS, CHANNEL_DATA_IDS, 
                    PATH_TO_PROCESSED_DATA)
from dataset import EventDataset as Dataset 


class StringListParamType(click.ParamType):
    name = 'string_list'

    def convert(self, value, param, ctx):
        if value is None:
            return []
        return value.split(',') 


@click.command("train")
@click.option("--dataset_dir", type=str, default=None)
@click.option("--dataset_file", type=str, default="dataset_events_-1.pickle")
@click.option("--batch_size", type=int, default=16)
@click.option("--num_workers", type=int, default=4)
@click.option("--weight_decay", type=float, default=0.0)
@click.option("--lr", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=2)
@click.option("--epochs", type=int, default=100)
@click.option("--mode", type=click.Choice(["pairwise", "leave_one_out"]), default="pairwise")
@click.option("--modality_types", type=StringListParamType(), default="respiratory,sleep_stages,ekg")
def train(
    dataset_dir,
    dataset_file,
    batch_size,
    num_workers,
    weight_decay, 
    lr, 
    lr_step_period,
    epochs,
    mode,
    modality_types
):
    if dataset_dir == None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    dataset_file_prefix = dataset_file.split(".")[0]
    modality_types_string = "_".join(modality_types)
    output = os.path.join(dataset_dir, f"outputs/output_{mode}_{dataset_file_prefix}_lr_{lr}_lr_sp_{lr_step_period}_wd_{weight_decay}_bs_{batch_size}_{modality_types_string}")

    output = os.path.join(CONFIG.OUTPUT, output)
    os.makedirs(output, exist_ok=True)
    temperature = torch.nn.parameter.Parameter(torch.as_tensor(0.))

    logger.info(f"modality_types: {modality_types}")
    logger.info(f"Path to dataset: {dataset_dir}")
    logger.info(f"Path to output: {output}")
    logger.info(f"Training Model: {mode}")

    logger.info(f"Batch Size: {batch_size}; Number of Workers: {num_workers}")
    logger.info(f"Weight Decay: {weight_decay}; Learning Rate: {lr}; Learning Step Period: {lr_step_period}")

    device = torch.device("cuda")
    logger.info(f"Device set to Cuda")

    num_targets = len(modality_types)
    ij = sum([((i, j), (j, i)) for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))], ())

    start = time.time()
    path_to_dataset = os.path.join(CONFIG.DATASETS, dataset_dir, dataset_file)
    dataset = {
        split: Dataset(path_to_dataset, 
                        split=split, 
                        modality_type=["respiratory", "sleep_stages", "ekg"])
        for split in ["pretrain", "valid"]
    }
    logger.info(f"Dataset loaded in {time.time() - start:.1f} seconds")

    model_dict = {}

    if "respiratory" in modality_types:
        model_resp = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["Respiratory"]), stride=2, dilation=1)
        model_resp.fc = torch.nn.Linear(model_resp.fc.in_features, 512)
        if device.type == "cuda":
            model_resp = torch.nn.DataParallel(model_resp)
        model_resp.to(device)
        model_dict["respiratory"] = model_resp

    if "sleep_stages" in modality_types:
        model_sleep = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["Sleep_Stages"]), stride=2, dilation=1)
        model_sleep.fc = torch.nn.Linear(model_sleep.fc.in_features, 512)
        if device.type == "cuda":
            model_sleep = torch.nn.DataParallel(model_sleep)
        model_sleep.to(device)
        model_dict["sleep_stages"] = model_sleep
    
    if "ekg" in modality_types:
        model_ekg = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["EKG"]), stride=2, dilation=1)
        model_ekg.fc = torch.nn.Linear(model_ekg.fc.in_features, 512)
        if device.type == "cuda":
            model_ekg = torch.nn.DataParallel(model_ekg)
        model_ekg.to(device)
        model_dict["ekg"] = model_ekg

    optim_params = []
    for model_key, model in model_dict.items():
        optim_params += list(model.parameters())

    optim_params.append(temperature)  # Append the temperature parameter

    optim = torch.optim.SGD(
        optim_params,
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    epoch_resume = 0
    best_loss = math.inf

    if os.path.isfile(os.path.join(output, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))

        if model_key == "respiratory":
            model_prefix = "resp"
        elif model_key == "sleep_stages": 
            model_prefix = "sleep"
        elif model_key == "ekg":
            model_prefix = "ekg"

        for model_key, model in model_dict.items():
            model.load_state_dict(checkpoint[f"{model_key}_state_dict"])

        # Loading temperature and other checkpointed parameters
        with torch.no_grad():
            temperature.fill_(checkpoint["temperature"])
        optim.load_state_dict(checkpoint["optim_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])

        # Other checkpointed values
        epoch_resume = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        logger.info(f"Resuming from epoch {epoch_resume}\n")
    else:
        logger.info("Starting from scratch")
    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    with open(os.path.join(output, "log", "{}.tsv".format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))), "w") as f:
        f.write("Epoch\tSplit\tTotal Loss\t")
        if mode == "pairwise":
            f.write("".join(f"{modality_types[i]}-{modality_types[j]} Loss\t" for (i, j) in ij))
            f.write("".join(f"{modality_types[i]}-{modality_types[j]} Accuracy\t" for (i, j) in ij))
        elif mode == "leave_one_out":
            f.write("".join(f"{modality_types[i]}-other Loss\tother-{modality_types[i]} Loss\t" for i in range(len(modality_types))))
            f.write("".join(f"{modality_types[i]}-other Accuracy\tother-{modality_types[i]} Accuracy\t" for i in range(len(modality_types))))
            
        f.write("Temperature\n")
        f.flush()
    
        for epoch in range(epoch_resume, epochs):
            for split in (["pretrain", "valid"] if epoch != -1 else ["valid"]):
                logger.info(f"Epoch: {epoch}; Split: {split}")
                dataloader = torch.utils.data.DataLoader(
                    dataset[split], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=(split == "train"))

                for model_key, model in model_dict.items():
                    model.train(split == "pretrain")

                if mode == "pairwise":
                    total_loss = 0.
                    total_pairwise_loss = np.zeros((num_targets, num_targets), dtype=float)
                    total_correct = np.zeros((num_targets, num_targets), dtype=int)
                    total_n = 0
                    total_pairs = np.zeros((num_targets, num_targets), dtype=int)
                elif mode == "leave_one_out":
                    total_loss = 0.
                    total_pairwise_loss = np.zeros((num_targets, 2), dtype=float)
                    total_correct = np.zeros((num_targets, 2), dtype=int)
                    total_n = 0
                    total_pairs = np.zeros((num_targets, 2), dtype=int)
                
                count = 0
                with torch.set_grad_enabled(split == "pretrain"):
                    with tqdm.tqdm(total=len(dataloader)) as pbar:
                        for (resp, sleep, ekg) in dataloader:

                            # if count == 10:
                            #     break
                            # count += 1

                            resp = resp.to(device, dtype=torch.float)
                            sleep = sleep.to(device, dtype=torch.float)
                            ekg = ekg.to(device, dtype=torch.float)

                            if len(modality_types) == 3:
                                emb = [
                                    model_dict["respiratory"](resp),
                                    model_dict["sleep_stages"](sleep),
                                    model_dict["ekg"](ekg),
                                ]
                            elif "respiratory" in modality_types and "sleep_stages" in modality_types:
                                emb = [
                                    model_dict["respiratory"](resp),
                                    model_dict["sleep_stages"](sleep),
                                ]
                            elif "respiratory" in modality_types and "ekg" in modality_types:
                                emb = [
                                    model_dict["respiratory"](resp),
                                    model_dict["ekg"](ekg),
                                ]
                            elif "sleep_stages" in modality_types and "ekg" in modality_types:
                                emb = [
                                    model_dict["sleep_stages"](sleep),
                                    model_dict["ekg"](ekg),
                                ]

                            for i in range(num_targets):
                                emb[i] = torch.nn.functional.normalize(emb[i])

                            if mode == "pairwise":
                                loss = 0.
                                pairwise_loss = np.zeros((num_targets, num_targets), dtype=float)
                                correct = np.zeros((num_targets, num_targets), dtype=int)
                                pairs = np.zeros((num_targets, num_targets), dtype=int)

                                for i in range(num_targets ):
                                    for j in range(i + 1, num_targets):

                                        logits = torch.matmul(emb[i], emb[j].transpose(0, 1)) * torch.exp(temperature)
                                        labels = torch.arange(logits.shape[0], device=device)
                            
                                        l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                                        loss += l
                                        pairwise_loss[i, j] = l.item()
                                        if len(logits) != 0:
                                            correct[i, j] = (torch.argmax(logits, axis=0) == labels).sum().item()
                                        else:
                                            correct[i, j] = 0
                                        pairs[i, j] = batch_size
                                        
                                        l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels.to(device), reduction="sum")
                                        loss += l
                                        pairwise_loss[j, i] = l.item()
                                        if len(logits) != 0:
                                            correct[j, i] = (torch.argmax(logits, axis=1) == labels).sum().item()
                                        else:
                                            correct[j, i] = 0
                                        pairs[j, i] = batch_size
                                loss /= len(ij)
                            if mode == "leave_one_out":
                                loss = 0.
                                pairwise_loss = np.zeros((num_targets, 2), dtype=float)
                                correct = np.zeros((num_targets, 2), dtype=int)
                                pairs = np.zeros((num_targets, 2), dtype=int)

                                for i in range(num_targets):
                                    other_emb = torch.stack([emb[j] for j in list(range(i)) + list(range(i + 1, num_targets))]).sum(0) / (num_targets - 1)
                                    logits = torch.matmul(emb[i], other_emb.transpose(0, 1)) * torch.exp(temperature)
                                    labels = torch.arange(logits.shape[0], device=device)
                            
                                    l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                                    loss += l
                                    pairwise_loss[i, 0] = l.item()
                                    if len(logits) != 0:
                                        correct[i, 0] = (torch.argmax(logits, axis=0) == labels).sum().item()
                                    else:
                                        correct[i, 0] = 0
                                    pairs[i, 0] = batch_size
                                    
                                    l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels.to(device), reduction="sum")
                                    loss += l
                                    pairwise_loss[i, 1] = l.item()
                                    if len(logits) != 0:
                                        correct[i, 1] = (torch.argmax(logits, axis=1) == labels).sum().item()
                                    else:
                                        correct[i, 1] = 0
                                    pairs[i, 1] = batch_size
                                loss /= num_targets * 2

                            total_loss += loss.item()
                            total_pairwise_loss += pairwise_loss
                            total_correct += correct
                            total_n += resp.shape[0]
                            total_pairs += pairs

                            loss /= resp.shape[0]
                            if split == "pretrain":
                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                            if temperature < 0:
                                with torch.no_grad():
                                    temperature.fill_(0)

                            if mode == "pairwise":
                                pbar.set_postfix_str(
                                    f"Loss: {total_loss / total_n:.5f} ({loss:.5f}); " +
                                    "Acc: {}; ".format(" ".join(map("{:.1f}".format, [100 * (total_correct[i, j] + total_correct[j, i]) / 2 / total_pairs[i, j] for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))]))) +
                                    f"Temperature: {temperature.item():.3f}"
                                )
                            elif mode == "leave_one_out":
                                pbar.set_postfix_str(
                                    f"Loss: {total_loss / total_n:.5f} ({loss:.5f}); " +
                                    "Acc: {}; ".format(" ".join(map("{:.1f}".format, [100 * (total_correct[i, 0] + total_correct[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1]) for i in range(len(modality_types))]))) +
                                    f"Temperature: {temperature.item():.3f}"
                                )
                            pbar.update()
                            
                if mode == "pairwise":
                    f.write("{}\t{}\t".format(epoch, split))
                    f.write(((len(ij) + 1) * "{:.5f}\t").format(total_loss / total_n, *[total_pairwise_loss[i, j] / total_pairs[i, j] for (i, j) in ij]))
                    f.write((len(ij) * "{:.3f}\t").format(*[100 * total_correct[i, j] / total_pairs[i, j] for (i, j) in ij]))
                    f.write("{:.5f}\n".format(temperature.item()))
                elif mode == "leave_one_out":
                    f.write("{}\t{}\t".format(epoch, split))
                    f.write(((num_targets  * 2 + 1) * "{:.5f}\t").format(total_loss / total_n, *[total_pairwise_loss[i, j] / total_pairs[i, j] for i in range(num_targets ) for j in [0, 1]]))
                    f.write(((num_targets  * 2) * "{:.3f}\t").format(*[100 * total_correct[i, j] / total_pairs[i, j] for i in range(num_targets ) for j in [0, 1]]))
                    f.write("{:.5f}\n".format(temperature.item()))
                f.flush()

            scheduler.step()

            loss = total_loss / total_n 
            is_best = (loss < best_loss)
            if is_best:
                best_loss = loss

            save = {
                "epoch": epoch,
                "temperature": temperature.item(),
                "optim_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "loss": loss
            }

            for model_key, model in model_dict.items():
                save[f"{model_key}_state_dict"] = model.state_dict()

            if is_best:
                torch.save(save, os.path.join(output, ".best.pt"))
                shutil.move(
                    os.path.join(output, ".best.pt"),
                    os.path.join(output, "best.pt")
                )
            torch.save(save, os.path.join(output, ".checkpoint.pt"))
            shutil.move(
                os.path.join(output, ".checkpoint.pt"),
                os.path.join(output, "checkpoint.pt")
            )


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def count_parameters(model):
    total_params = 0
    total_layers = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        total_layers += 1
        # print(f"Layer: {name}, Shape: {param.shape}")

    return total_layers, total_params

if __name__ == '__main__':
    train()
