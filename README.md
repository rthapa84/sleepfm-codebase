# SleepFM (Sleep Foundation Model)

## 🔥 News
- SleepFM was accepted at ICML 2024!
- Shorter version of our paper acceped to ICLR TS4H workshop and AAAI 2024 SSS on Clinical FMs
- [Our paper](https://arxiv.org/abs/2405.17766v1) is out on arxiv.

## 📖 Introduction
Sleep is a complex physiological process evaluated through various modalities recording electrical brain, cardiac, and respiratory activities. We curate a large polysomnography dataset from over 14,000 participants comprising over 100,000 hours of multi-modal sleep recordings. Leveraging this extensive dataset, we developed SleepFM, the first multi-modal foundation model for sleep analysis. We show that a novel leave-one-out approach for contrastive learning significantly improves downstream task performance compared to representations from standard pairwise contrastive learning. A logistic regression model trained on SleepFM's learned embeddings outperforms an end-to-end trained convolutional neural network (CNN) on sleep stage classification (macro AUROC 0.88 vs 0.72 and macro AUPRC 0.72 vs 0.48) and sleep disordered breathing detection (AUROC 0.85 vs 0.69 and AUPRC 0.77 vs 0.61).  Notably, the learned embeddings achieve 48% top-1 average accuracy in retrieving the corresponding recording clips of other modalities from 90,000 candidates. This work demonstrates the value of holistic multi-modal sleep modeling to fully capture the richness of sleep recordings.


# 📖 Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Licence](#license)

<a name="installation"/>

# 💿 Installation

Please use the following steps to create an environment for running SleepFM

```bash
git clone https://github.com/rthapa84/sleepfm-codebase.git
cd sleepfm-codebase
conda env create -f environment.yml
conda activate sleepfm_env
```

<a name="usage"/>

# 👩‍💻 Usage

*This is a research code. Here, we provide our pretraining pipeline with a publicly available dataset, as we cannot release our internal pretraining dataset at the moment.*

This codebase will serve as a framework that you can adapt to your dataset for pretraining and testing. Below, we outline the steps to pretrain and adapt the model on a publicly available dataset called [The PhysioNet/Computing in Cardiology Challenge 2018 (CinC)](https://physionet.org/content/challenge-2018/1.0.0/test/#files-panel). Please keep in mind that this dataset is small and will most likely not yield optimal results.

## Downloading Dataset

We are working with CinC dataset as a dummy usecase. 

- Follow the [link](https://physionet.org/content/challenge-2018/1.0.0/test/#files-panel). You may need to create a physionet account. 
- Scroll to the bottom where you will see Files, and either directly download as zip files or run the following command. `wget -r -N -c -np https://physionet.org/files/challenge-2018/1.0.0/`

Now, your data is downloaded to the server. Later on, we will use this path to reference the dataset.

All our main pipeline Python scripts are inside the `sleepfm/` folder. The numbers following the scripts indicate the order in which they are intended to be run. We also provide corresponding bash scripts to execute the Python scripts inside the `sleepfm/bash_scripts` folder. Below, we describe each step.

Also note that there are `sleepfm/utils.py` and `sleepfm/config.py` helper scripts. The `utils.py` script contains all the helper functions and `config.py` contains all the paths and global variables used at different steps in the pipeline below. Make sure to take a look at the file and set the paths according to your needs. 

## Preprocessing Dataset

PSG files may be stored in different formats. Here, we specifically provide scripts to process .EDF file format. PSG events can also be stored in multiple different formats. We provide preprocessing scripts for CinC, but please note that you will more than likely need to change these initial preprocessing scripts, which are used to extract the 30-second epoch data from EDF files and their corresponding labels from the events file. This step does not require GPU support. 

- **Step 1:** `0_extract_pretraining_data.py`
  - This script converts the PSG files saved in the raw data folder to short 30-second epoch `.npy` files. It extracts the necessary channels and sleep-related events as well. 

- **Step 2:** `1_prepare_dataset.py`
  - This script creates the pretrain/train/valid/test split and manages them inside pickle files to be used later during pretraining and evaluation. 

## Pretraining

SleepFM uses 1D CNN and contrastive learning for pretraining. For more details about our model architecture, please check out our [paper](https://arxiv.org/abs/2405.17766v1). This step requires GPU support. 

- **Step 1:** `2_pretrain.py`
  - This script trains our contrastive learning framework using all three modalities (respiratory, sleep stages, and EKG). Note that we call the brain activity signal modality sleep stages here because they are primarily used for sleep staging. 
- **Step 2:** `3_generate_embed_pretraining.py`
  - After pretraining our model, we want to generate the embeddings for train/valid/test so that we can train a linear head (logistic regression) for downstream classification. We do sleep stage classification here. 

## Evaluation

Note: These evaluation results will not match the ones that we have in our paper. We cannot release our dataset at the moment. This step does not require GPU support. 

- **Step 2:** `4_classification_eval_pretraining.py`
  - Finally, this script trains a logistic regression model and calculates performance metrics such as AUROC and AUPRC.


## Model Checkpoint

We provide one of our model checkpoints inside the `sleepfm/checkpoint` folder. You can load the model as shown in the `sleepfm/3_generate_embed_pretraining.py` script. Follow all the other steps, but skip the `sleepfm/2_pretrain.py` step if you use this checkpoint. Ensure that you set the paths correctly in the `sleepfm/config.py` file.

**Note that this is a really small model. We are currently working on a larger version with some architectural improvements and trained on more data. We will be releasing the codebase and model for that soon as well. Stay tuned!👀**

## BibTeX

```bibtex
@inproceedings{thapa2024sleepfm,
  title={SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals},
  author={Rahul Thapa and Bryan He and Magnus Ruud Kjaer and Hyatt Moore and Gauri Ganjoo and Emmanuel Mignot and James Zou},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## 🪪 License

[MIT License](LICENSE)
