import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
import operator
from TVB.logger import logger
from tqdm import tqdm
try:
    from collections import Counter
except:
    from collections.abc import Counter



def interpolate(data, freq_from, freq_to=30):
    interpolated_data = {}
    for exp_key in data:
        interpolated_data_ = {}
        for filename in data[exp_key]:
            filedata, filelabel = data[exp_key][filename]
            n_current = len(filedata)
            x_current = np.arange(0, n_current)
            num_samples = np.floor((n_current/freq_from)*freq_to).astype(int)
            x_required = np.arange(0, num_samples)
            interpolated_data_[filename] = [np.interp(x_required, x_current, filedata), filelabel]
        interpolated_data[exp_key] = interpolated_data_
    return interpolated_data



def make_numpyloader(data, window_size, stride, normalisation):
    chunked_data = {}
    for exp_key in data:
        chunked_data_ = {}
        for filename in data[exp_key]:
            data_, label_ = data[exp_key][filename]
            chunk_data, chunk_label = chunks(data_, label_, window_size, stride, normalisation)
            chunked_data_[filename] = [chunk_data, chunk_label]
        chunked_data[exp_key] = chunked_data_
    return chunked_data

def make_dataloaders(data, label, window_size, stride, normalisation, batch_size, shuffle):
    data_, labels = [], []
    for idx in tqdm(range(len(data))):
        data_chunks, label_chunks = chunks(data[idx], label[idx], window_size, stride, normalisation)
        data_.append(data_chunks)
        labels.append(label_chunks)
    dataset = TensorDataset(torch.from_numpy(np.array(data_)), torch.from_numpy(np.array(labels)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def normalise_data(data, norm_type='minmax'):
    if norm_type=='z':
        data = (data-np.mean(data))/np.std(data)
    elif norm_type=='minmax':
        data = (data - min(data))/(max(data)-min(data))
    else:
        raise NotImplementedError
    return data

def chunks(signal, label_, window_size, stride, normalisation):
    data, label = [], []
    n_chunks = int((len(signal)-window_size)/stride) +  1
    for i in range(n_chunks):
        chunk = signal[int(i*stride) : int(window_size+(i*stride))]
        chunk = normalise_data(chunk, norm_type=normalisation)
        data.append(chunk)
        label.append(label_)
    return data, label

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

def pred_from_dict(data, model, device, window_size, tensordataset, batch_size,
                    num_workers, shuffle, progressbar):
    raiseException(tensordataset)
    logger.info('Loading dict data into torch tensordataset. Avoid this on large datasets..')
    model.eval().float().to(device)
    exp_results = {}
    for exp_key in data:
        file_results = {}
        for filename in data[exp_key]:
            file_data, file_label = data[exp_key][filename]
            file_data = np.array(file_data)
            # print(file_data)
            # print(file_data.shape)
            try: 
                assert len(file_data.shape) == 2
            except:
                logger.info(f"No windows found, skipping file {filename} in {exp_key}")
                continue
            results = {'predicted_class':[], 'posterior0':[], 'posterior1':[]}
            tensordataset = TensorDataset(torch.from_numpy(file_data))
            dataloader = DataLoader(tensordataset, batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=shuffle)
            for data_ in tqdm(dataloader, disable=(not progressbar)):
                data_ = data_[0].unsqueeze(-1).float()
                out = model(data_.to(device))
                results_ = model.results(out)
                #refactor
                for key in results_:
                    res = np.array(results_[key])
                    expand_dims = True if res.shape[0] == 1 else False
                    res = res.squeeze()
                    if expand_dims:
                        res = np.expand_dims(res, 0)
                    results_[key] = res
                {key:results[key].extend(results_[key]) for key in results}

            for key in results:
                res = np.array(results[key]).flatten()
                results[key] = res
            file_results[filename] = results
        exp_results[exp_key] = file_results
    return exp_results
