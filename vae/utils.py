import torch
import os
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def partition_dataset(dataset, size):
    dataloader = DataLoader(dataset, batch_size=size, shuffle=True)
    x, y = [d.to(device) for d in next(iter(dataloader))]
    return TensorDataset(x, y)

def create_dataloader(tensor, size, batch_size):
    dataset = TensorDataset(tensor, torch.zeros(size).to(device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def combine_dataloader(tensor_a, dataset_b, size, batch_size, ab_ratio):
    dataset_a = TensorDataset(tensor_a, torch.zeros(size).to(device))
    an = int(ab_ratio * size)
    bn = size - an
    dataset_a = partition_dataset(dataset_a, an)
    dataset_b = partition_dataset(dataset_b, bn) 
    dataset_c = ConcatDataset([dataset_a, dataset_b])
    return DataLoader(dataset_c, batch_size=batch_size, shuffle=True)

def load_model(path):
    if os.path.isfile(path):
        return torch.load(path, weights_only=False)
    else:
        raise Exception("Initialize the model first with 'train.py'")