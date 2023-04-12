import torch
import os
import torch.nn as nn
import random
import numpy as np

def check_model_overwrite(args):
    if not args['overwrite_model']:
        if os.path.exists(args['save_chk_name']):
            raise Exception(f'model {args["save_chk_name"]} already exists')

def get_output_metrics(output, label, metrics, key, thres=0.5):
    preds = ((torch.sigmoid(output)>thres).float() == label).float().detach().cpu() .tolist()
    if isinstance(preds, list):
        metrics[key].extend(preds)
    else:
        metrics[key].append(preds)
    # exit()
def train_model(model, loaders, device, training_args):
    check_model_overwrite(training_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args['lr']) 
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    train_loader, val_loader = loaders
    metrics = {'train_all':[], 'val_all':[], 'train_epoch':[], 'val_epoch':[], 'best_val_loss':np.inf, 
               'all_train_preds':[], 'all_val_preds':[], 'train_accuracy':[], 'val_accuracy':[]}
    for idx in range(training_args['epochs']):
        model.train()
        track_loss = []
        for data, label in train_loader:
            data = data.squeeze()
            data, label = check_and_make_segments(data, label.squeeze(), training_args)
            data = data.to(device).squeeze()
            label = label.to(device)
            output = model(data.unsqueeze(-1).float())
            label = label.repeat(10).float()
            loss = criterion(output.squeeze(), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            get_output_metrics(output.squeeze(), label, metrics, key='all_train_preds')
            track_loss.append(loss.item())
        metrics['train_all'].extend(track_loss)
        metrics['train_epoch'].append(sum(track_loss)/len(track_loss))
        metrics['train_accuracy'].append(sum(metrics['all_train_preds'])/len(metrics['all_train_preds']))
        metrics['all_train_preds'] = []
        model.eval()
        
        data = data.squeeze()
        data, label = check_and_make_segments(data, label.squeeze(), training_args)
        track_loss = []
        for data, label in val_loader:
            data = data.to(device).squeeze()
            label = label.to(device)
            output = model(data.unsqueeze(-1).float())
            label = label.squeeze().repeat(10).float()
            loss = criterion(output.squeeze(), label)
            track_loss.append(loss.item())
            get_output_metrics(output.squeeze(), label, metrics, key='all_val_preds')
        metrics['val_all'].extend(track_loss)
        metrics['val_epoch'].append(sum(track_loss)/len(track_loss))
        metrics['val_accuracy'].append(sum(metrics['all_val_preds'])/len(metrics['all_val_preds']))
        metrics['all_val_preds'] = []
        if metrics['val_epoch'][-1] < metrics['best_val_loss']:
            metrics['best_val_loss'] = metrics['val_epoch'][-1]
            if training_args['save_by_validation_chk']:
                torch.save(model.state_dict(), training_args['save_chk_name'])
        model.eval()
        if training_args['metrics_verbose'] == 1:
            print(f'{idx}\t{round(metrics["train_epoch"][-1], 4)}\t{round(metrics["val_epoch"][-1], 4)}\t{round(metrics["train_accuracy"][-1], 4)}\t{round(metrics["val_accuracy"][-1], 4)}')
            
def check_and_make_segments(data, label, training_args):
    if 'random_segment_training' not in training_args: return data
    if training_args['random_segment_training'] == False: return data
    assert 'min_segment_length' in training_args, f'sepcify menimum length for segments with "min_segment_length"'
    seglen = training_args['min_segment_length']
    if len(data)>seglen:
        size = random.choice([i for i in range(seglen, len(data))])
        start = random.choice([i for i in range(0, len(data)-size+1)])
        end = start + size
    else:
        start, end = 0, len(data)
    data = data[start:end].float()
    label = label[start:end]
    return data, label