import torch
import os
import numpy as np
from CaMLsort.logger import logger
from torch.utils.data import Dataset
from CaMLsort.tvb_utils import *
from huggingface_hub import hf_hub_download
import shutil
import yaml
from attrdict import AttrDict
from pathlib import Path
from scipy.io import loadmat

def read_yaml(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg

class TorchDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass
    
    def __len__(self):
        return
    
    def __call__(self):
        return

def download_sample_dataset(path, repo_id, name="sample_dataset.zip"):
    if os.path.exists(os.path.join(path, name)):
        pass
    else:
        logger.info(f'Downloading sample dataset')
        hf_hub_download(
                        repo_id=repo_id,
                        filename="Sample Data.zip",
                        cache_dir=path,
                        force_filename=name,
                        )
    shutil.unpack_archive(os.path.join(path, name), path)
    return load_sample_dataset(os.path.join(path, "Sample Data"))

def load_sample_dataset(path, signal_key="calcium", label_key="state"):
    all_data_dict = {}
    for setname in  os.listdir(path):
        set_dict = {}
        for matfile in os.listdir(os.path.join(path, setname)):
            matfile = os.path.join(path, setname, matfile)
            full_data  = loadmat(matfile)    
            data = full_data[signal_key].squeeze()
            label = full_data[label_key].squeeze()
            set_dict[Path(matfile).stem] = [data, label]
        all_data_dict[setname] = set_dict
    return all_data_dict

def read_data(data, label, filename, exp_name):
    if isinstance(data, list):
        data = np.array(data)
  
    if isinstance(data, np.ndarray):
        data_dict = {}
        data_dict[exp_name] = {}
        if filename is not None:
            assert len(filename) == len(data)
        else:
            filename = [str(i) for i in range(len(data))]
        for i in range(len(data)):
            data_dict[exp_name][filename[i]] = [data[i], label[i] if label is not None else None]
        data = data_dict
    elif isinstance(data, dict):
        for key in data:
            assert isinstance(data[key], dict)
    else:
        raise NotImplementedError

    return data

def check_sampling_rate(data, sampling_rate):
        if sampling_rate == 30:
            logger.info("Assuming data is processed at 30Hz")
        else:
            logger.info(f"Interpolating data from {sampling_rate}Hz to 30Hz")
            data = interpolate(data, sampling_rate)
        return data