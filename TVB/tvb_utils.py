import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def read_data(data):
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, list):
        data = np.array(data)

    else:
        raise NotImplementedError

    if len(data.shape) >2:
        raise NotImplementedError
    return data

def make_dataloaders(data, window_size, stride, normalisation):
    data_ = []
    if len(data.shape) == 2:
        for idx in range(len(data)):
            data_.append(chunks(data[idx], window_size, stride, normalisation))
    else:
        raise NotImplementedError
    dataset = TensorDataset(np.array(data_))
    dataloader = DataLoader(dataset)
    return dataloader

def normalise_data(data, norm_type='minmax'):
    if norm_type=='z':
        data = (data-np.mean(data))/np.std(data)
    elif norm_type=='minmax':
        data = (data - min(data))/(max(data)-min(data))
    else:
        raise NotImplementedError
    return data

def chunks(signal, window_size, stride, normalisation):
    data = []
    n_chunks = int((len(signal)-window_size)/stride) +  1
    for i in range(n_chunks):
        chunk = signal[int(i*stride) : int(window_size+(i*stride))]
        chunk = normalise_data(chunk, norm_type=normalisation)
        data.append(chunk)
    return data
