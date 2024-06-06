import pandas as pd
from tqdm import tqdm
import pickle
import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter

import sys
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

import sys
sys.path.append("../model")
import models
import config
from config import (MODALITY_TYPES, CLASS_LABELS, 
                    LABELS_DICT, PATH_TO_PROCESSED_DATA)
from utils import train_model
from dataset import EventDataset as Dataset 


def main(args):

    dataset_dir = args.dataset_dir

    if dataset_dir == None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    output_file = args.output_file
    path_to_output = os.path.join(dataset_dir, f"{output_file}")
    breakpoint()
    modality_type = args.modality_type
    num_per_event = args.num_per_event
    model_name = args.model_name

    path_to_figures = os.path.join(path_to_output, f"figures")
    path_to_models = os.path.join(path_to_output, f"models")
    path_to_probs = os.path.join(path_to_output, f"probs")

    os.makedirs(path_to_figures, exist_ok=True)
    os.makedirs(path_to_models, exist_ok=True)
    os.makedirs(path_to_probs, exist_ok=True)

    dataset_file = "dataset.pickle"
    dataset_event_file = "dataset_events_-1.pickle"

    test_emb_file = f"dataset_events_-1_test_emb.pickle"
    valid_emb_file = f"dataset_events_-1_valid_emb.pickle"
    train_emb_file = f"dataset_events_-1_train_emb.pickle"

    logger.info(f"modality_type: {modality_type}")
    logger.info(f"dataset_file: {dataset_file}")
    logger.info(f"dataset_event_file: {dataset_event_file}")
    logger.info(f"test_emb_file: {test_emb_file}")
    logger.info(f"valid_emb_file: {valid_emb_file}")

    path_to_dataset = os.path.join(dataset_dir, f"{dataset_file}")
    with open(path_to_dataset, "rb") as f:
        dataset = pickle.load(f)

    path_to_event_dataset = os.path.join(dataset_dir, f"{dataset_event_file}")
    with open(path_to_event_dataset, "rb") as f:
        dataset_events = pickle.load(f)

    path_to_eval_data = os.path.join(path_to_output, f"eval_data")
    with open(os.path.join(path_to_eval_data, test_emb_file), "rb") as f:
        emb_test = pickle.load(f)

    with open(os.path.join(path_to_eval_data, valid_emb_file), "rb") as f:
        emb_valid = pickle.load(f)

    with open(os.path.join(path_to_eval_data, train_emb_file), "rb") as f:
        emb_train = pickle.load(f)

    path_to_label = {}

    for split, split_dataset in tqdm(dataset.items()):
        for patient_data in tqdm(split_dataset):
            mrn = list(patient_data.keys())[0]
            for event, event_paths in patient_data[mrn].items():
                for event_path in event_paths:
                    path_to_label[event_path] = event 

    labels_test = np.array([path_to_label[event_path[0]] for event_path in dataset_events["test"]])
    labels_valid = np.array([path_to_label[event_path[0]] for event_path in dataset_events["valid"]])
    labels_train = np.array([path_to_label[event_path[0]] for event_path in dataset_events["train"]])

    counter_test = Counter(labels_test)
    counter_valid = Counter(labels_valid)
    counter_train = Counter(labels_train)

    logger.info(f"Test Labels: {counter_test}")
    logger.info(f"Valid Labels: {counter_valid}")
    logger.info(f"Train Labels: {counter_train}")

    indices_train = []
    for i in range(len(labels_train)):
        if labels_train[i] in LABELS_DICT:
            indices_train.append(i)

    indices_test = []
    for i in range(len(labels_test)):
        if labels_test[i] in LABELS_DICT:
            indices_test.append(i)

    if modality_type == "combined":
        emb_test = np.concatenate(emb_test, axis=1)
        emb_valid = np.concatenate(emb_valid, axis=1)
        emb_train = np.concatenate(emb_train, axis=1)
    else:
        target_index = MODALITY_TYPES.index(modality_type)
        emb_test = emb_test[target_index]
        emb_valid = emb_valid[target_index]
        emb_train = emb_train[target_index]

    X_train = emb_train[indices_train]
    y_train = np.array(labels_train)[indices_train]
    y_train = np.array([LABELS_DICT[item] for item in y_train])

    X_test = emb_test[indices_test]
    y_test = np.array(labels_test)[indices_test]
    y_test = np.array([LABELS_DICT[item] for item in y_test])

    path_to_save = os.path.join(path_to_figures, f"{model_name}_{modality_type}")
    model, y_probs, class_report = train_model(X_train, X_test, y_train, y_test, path_to_save, 
                list(CLASS_LABELS), model_name=model_name, max_iter=args.max_iter)

    logger.info(f"Saving model...")
    with open(os.path.join(path_to_models, f"{modality_type}_model.pickle"), 'wb') as file:
        pickle.dump(model, file)

    logger.info(f"Saving probabilities...")
    with open(os.path.join(path_to_probs, f"{modality_type}_y_probs.pickle"), 'wb') as file:
        pickle.dump(y_probs, file)

    logger.info(f"Saving class report...")
    with open(os.path.join(path_to_probs, f"{modality_type}_class_report.pickle"), 'wb') as file:
        pickle.dump(class_report, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files and generate plots.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file name")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to preprocessed data")
    parser.add_argument("--modality_type", type=str, help="Target Types", choices=["respiratory", "sleep_stages", "ekg", "combined"], default="combined")
    parser.add_argument("--num_per_event", type=int, default=-1, help="Number of events from each event group")
    parser.add_argument("--model_name", type=str, default="logistic", choices=["logistic", "xgb"], help="Type of model")
    parser.add_argument("--max_iter", type=int, default=100, help="Max Iter for LR model")

    args = parser.parse_args()
    main(args)
