import argparse
import importlib
import json
import sys

import torch
import torch.nn as nn

from datasets import link_prediction
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
import models

def get_agg_class(agg_class):
    """
    Parameters
    ----------
    agg_class : str
        Name of the aggregator class.

    Returns
    -------
    layers.Aggregator
        Aggregator class.
    """
    return getattr(sys.modules[__name__], agg_class)

def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.

    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'link_prediction':
        criterion = nn.BCEWithLogitsLoss()

    return criterion

def get_dataset(args):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    task, dataset_name, *dataset_args = args
    if task == 'link_prediction':
        class_attr = getattr(importlib.import_module('datasets.link_prediction'), dataset_name)
        dataset = class_attr(*dataset_args)

    return dataset

def get_fname(config):
    """
    Parameters
    ----------
    config : dict
        A dictionary with all the arguments and flags.

    Returns
    -------
    fname : str
        The filename for the saved model.
    """
    duplicate_examples = config['duplicated_examples']
    repeat_examples = config['repeat_examples']
    agg_class = config['agg_class']
    hidden_dims_str = '_'.join([str(x) for x in config['hidden_dims']])
    num_samples = config['num_samples']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    fname = 'graphsage_agg_class_{}_hidden_dims_{}_num_samples_{}_batch_size_{}_epochs_{}_lr_{}_weight_decay_{}_duplicate_{}_repeat_{}.pth'.format(
        agg_class, hidden_dims_str, num_samples, batch_size, epochs, lr,
        weight_decay, duplicate_examples, repeat_examples)

    return fname

# def get_model(config):
#     """
#     Parameters
#     ----------
#     config : dict
#         A dictionary with all the arguments and flags.
# 
#     Returns
#     -------
#     model : torch.nn.Module
#         The model.
#     """
#     if config['model'] == 'GraphSAGE':
#        agg_class = get_agg_class(config['agg_class'])
#        model = models.GraphSAGE(config['input_dim'], config['hidden_dims'],
#                                 config['output_dim'], config['dropout'],
#                                 agg_class, config['num_samples'],
#                                 config['device'])
#     elif config['model'] == 'GAT':
#         model = models.GAT(config['input_dim'], config['hidden_dims'],
#                            config['output_dim'], config['num_heads'],
#                            config['dropout'], config['device'])
# 
#     model.to(config['device']) 
# 
#     return model


def parse_args():
    """
    Returns
    -------
    config : dict
        A dictionary with the required arguments and flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', type=str, default='config.json',
                        help='path to json file with arguments, default: config.json')

    parser.add_argument('--stats_per_batch', type=int, default=16,
                        help='print loss and accuracy after how many batches, default: 16')

    parser.add_argument('--dataset', type=str,
                        choices=['CollegeMsg', 'BitcoinAlpha'],
                        required=True,
                        help='name of the dataset')
    parser.add_argument('--dataset_path', type=str,
                        required=True,
                        help='path to dataset')
    parser.add_argument('--neg_examples_path', type=str,
                        default='',
                        help='path to file with negative examples, default=''')
    parser.add_argument('--duplicate_examples', action='store_true',
                        help='whether to allow duplicate edges in the list of positive examples, default=False')
    parser.add_argument('--repeat_examples', action='store_true',
                        help='whether to use positive examples that have already been seen during graph construction or training, default=False')
    parser.add_argument('--self_loop', action='store_true',
                        help='whether to add self loops to adjacency matrix, default=False')
    parser.add_argument('--normalize_adj', action='store_true',
                        help='whether to normalize adj like in gcn, default=False')

    parser.add_argument('--task', type=str,
                        choices=['unsupervised', 'link_prediction'],
                        default='link_prediction',
                        help='type of task, default=link_prediction')

    parser.add_argument('--agg_class', type=str,
                        choices=[MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator],
                        default=MaxPoolAggregator,
                        help='aggregator class, default: MaxPoolAggregator')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use GPU, default: False')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout out, currently only for GCN, default: 0.5')
    parser.add_argument('--hidden_dims', type=int, nargs="*",
                        help='dimensions of hidden layers, length should be equal to num_layers, specify through config.json')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of neighbors to sample, default=-1')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size, default=32')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs, default=2')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, default=1e-4')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay, default=5e-4')

    parser.add_argument('--save', action='store_true',
                        help='whether to save model in trained_models/ directory, default: False')
    parser.add_argument('--load', action='store_true',
                        help='whether to load model in trained_models/ directory')

    args = parser.parse_args()
    config = vars(args)
    if config['json']:
        with open(config['json']) as f:
            json_dict = json.load(f)
            config.update(json_dict)

    config['num_layers'] = len(config['hidden_dims']) + 1

    print('--------------------------------')
    print('Config:')
    for (k, v) in config.items():
        print("    '{}': '{}'".format(k, v))
    print('--------------------------------')

    return config