import json
import gzip
import os
from torch.utils import data
import random
import torch

class RedditDataset:
    """
    A class that handles the reddit data files
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        while True:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)

                    if "response" in data and "context" in data:
                        yield [data["response"], data["context"]]

class Dataset:
    """
    A class that handles one dataset
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        max_dataset_size = 10*1000*1000    #Cache small datasets in memory
        dataset = []
        data_format = None

        while dataset is None or len(dataset) == 0:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        try:
                            data = data['texts']
                        except:
                            continue
                    if data_format is None:
                        data_format = len(data)
                    
                    #Ensure that all entries are of the same 2/3 col format
                    assert len(data) == data_format

                    if dataset is not None:
                        dataset.append(data)
                        if len(dataset) >= max_dataset_size:
                            dataset = None

                    yield data
                
        # Data loaded. Now stream to the queue
        # Shuffle for each epoch
        while True:
            random.shuffle(dataset)
            for data in dataset:
                yield data
       
class DatasetForDataLoader(data.Dataset):
    def __init__(self, datasets, batch_size, dataset_indices):
        self.datasets = datasets
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
    
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        data_idx = random.choice(self.dataset_indices)
        return next(self.datasets[data_idx])

def get_data_paths_and_indices(config):
    with open(config["data_config"]) as fIn:
        data_config = json.load(fIn)

    with open(config["num_cols_config"]) as fIn:
        num_cols = json.load(fIn)

    filepaths_cols2 = []
    dataset_indices_cols2 = []

    filepaths_cols3 = []
    dataset_indices_cols3 = []

    k = 0
    for _, data in enumerate(data_config):
        if data['name'] in num_cols and num_cols[data['name']] == 2:
            filepaths_cols2.append(os.path.join(os.path.expanduser(config["data_folder"]), data['name']))
            dataset_indices_cols2.extend([k]*data['weight'])
            k += 1
    
    k = 0
    for _, data in enumerate(data_config):
        if data['name'] in num_cols and num_cols[data['name']] == 3:
            filepaths_cols3.append(os.path.join(os.path.expanduser(config["data_folder"]), data['name']))
            dataset_indices_cols3.extend([k]*data['weight'])
            k += 1

    return filepaths_cols2, dataset_indices_cols2, filepaths_cols3, dataset_indices_cols3


def prep_dataset(filepaths, dataset_indices):
    datasets = []
    for filepath in filepaths:
        if "reddit_" in filepath:       #Special dataset class for Reddit files
            data_obj = RedditDataset(filepath)
        else:
            data_obj = Dataset(filepath)
        datasets.append(iter(data_obj)) 

    return datasets

def get_dataset_for_dataloaders(config, batch_size):
    print("Starting data loading...")
    filepaths_cols2, dataset_indices_cols2, filepaths_cols3, dataset_indices_cols3 = get_data_paths_and_indices(config)
    
    datasets_cols2 = prep_dataset(filepaths_cols2, dataset_indices_cols2)
    datasets_cols3 = prep_dataset(filepaths_cols3, dataset_indices_cols3)

    datasets_for_dataloader_cols2 = DatasetForDataLoader(datasets_cols2, batch_size, dataset_indices_cols2)
    datasets_for_dataloader_cols3 = DatasetForDataLoader(datasets_cols3, batch_size, dataset_indices_cols3)

    return datasets_for_dataloader_cols2, datasets_for_dataloader_cols3, len(dataset_indices_cols2), len(dataset_indices_cols3)
                      
