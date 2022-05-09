import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import operator
from tqdm import tqdm
from collections import Counter

def read_data(data):
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, list):
        data = np.array(data)

    else:
        raise NotImplementedError

    raiseException(len(data.shape), '<=', 2)
    return data

def make_numpyloader(data, window_size, stride, normalisation):
    data_ = []
    if len(data.shape) == 2:
        for idx in range(len(data)):
            data_.append(chunks(data[idx], window_size, stride, normalisation))
    else:
        raise NotImplementedError
    return np.array(data_)

def make_dataloaders(data, window_size, stride, normalisation):
    data_ = []
    if len(data.shape) == 2:
        for idx in range(len(data)):
            data_.append(chunks(data[idx], window_size, stride, normalisation))
    else:
        raise NotImplementedError
    dataset = TensorDataset(torch.from_numpy(np.array(data_)))
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

def raiseException(data, condition='==', value=True, exception=NotImplementedError):
    operator_mappings = {
                        '<': operator.lt,
                        '>=': operator.ge,
                        '==': operator.eq,
                        '>': operator.gt,
                        '<=': operator.le,
                        '!=': operator.ne,
                        }

    try:
        assert operator_mappings[condition](data,value)
    except:
        raise exception


def pred_from_numpy(data, model, device, window_size, tensordataset, batch_size,
                    collate_fn, num_workers, shuffle, progressbar):
    raiseException(tensordataset)
    model.eval().float().to(device)
    outputs = []
    assert len(data.shape) in [2, 3]
    if len(data.shape) == 2: data = np.expand_dims(data, 0)
    all_results = []
    for idx in range(len(data)):
        results = {'class':[], 'post0':[], 'post1':[]}
        if tensordataset:
            tensordataset = TensorDataset(torch.from_numpy(data[idx]))
            dataloader = DataLoader(tensordataset, batch_size=batch_size,
                                    collate_fn=collate_fn, num_workers=num_workers,
                                    shuffle=shuffle)
            for data_ in tqdm(dataloader, disable=(not progressbar)):
                data_ = data_[0].unsqueeze(-1).float()
                out = model(data_.to(device))
                results_ = model.results(out)
                {k:results[k].extend(results_[k]) for k in results}
        all_results.append(results)
    return all_results
