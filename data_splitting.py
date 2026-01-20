# Created by julia at 30.09.2024
import numpy as np
import torch
import math
import pickle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def split(config, seed=4242):
    # Datasplits, inner and outer CV + inner random subsplit for ES
    np.random.seed(seed)  # normally default, but for test run other seed for splitting!

    # Get y values (for stratified splits)
    with open(config['data_file'], "rb") as f:
        data = pickle.load(f)
    y = np.empty(len(data))
    for i in range(len(data)):
        y[i] = data[i].y

    # Precompute split sizes
    # Size in CV may differ from fold to fold! ->max size computed, arrays filled with neg. values, removed later
    num_outer_test_samples = math.ceil(config['split_test_fraction'] * config['number_samples'])
    num_outer_train_val_samples = config['number_samples'] - num_outer_test_samples

    num_max_inner_train_val_samples = math.ceil(num_outer_train_val_samples * (1 - (1 / config['cv_fold_size'])))
    num_max_inner_test_samples = math.ceil(num_outer_train_val_samples * (1 / config['cv_fold_size']))
    num_max_inner_val_samples = math.ceil(num_max_inner_train_val_samples * config['split_early_stopping_fraction'])
    num_max_inner_train_samples = num_max_inner_train_val_samples - num_max_inner_val_samples
    num_max_outer_val_samples = math.ceil(num_outer_train_val_samples * (1 / config['cv_fold_size_es_outer']))
    num_max_outer_train_samples = math.ceil(num_outer_train_val_samples * (1 - (1 / config['cv_fold_size_es_outer'])))

    # Create split indices
    idx = np.arange(config['number_samples'])
    outer_test_idx = np.empty((config['number_train_test_splits'], num_outer_test_samples), dtype=int)
    outer_train_val_idx = np.empty((config['number_train_test_splits'], num_outer_train_val_samples), dtype=int)
    inner_test_idx = np.full((config['number_train_test_splits'], config['cv_fold_size'], num_max_inner_test_samples),
                             -1, dtype=int)
    inner_val_idx = np.full((config['number_train_test_splits'], config['cv_fold_size'], num_max_inner_val_samples), -1,
                            dtype=int)
    inner_train_idx = np.full((config['number_train_test_splits'], config['cv_fold_size'], num_max_inner_train_samples),
                              -1, dtype=int)
    outer_train_idx = np.full(
        (config['number_train_test_splits'], config['cv_fold_size_es_outer'], num_max_outer_train_samples), -1,
        dtype=int)
    outer_val_idx = np.full(
        (config['number_train_test_splits'], config['cv_fold_size_es_outer'], num_max_outer_val_samples), -1,
        dtype=int)

    # Define Splitting Methods
    random_split_outer = StratifiedShuffleSplit(n_splits=config['number_train_test_splits'],
                                                test_size=config['split_test_fraction'])
    kf_split_inner = StratifiedKFold(n_splits=config['cv_fold_size'])
    random_split_inner = StratifiedShuffleSplit(n_splits=1, test_size=config['split_early_stopping_fraction'])
    kf_split_outer = StratifiedKFold(n_splits=config['cv_fold_size_es_outer'])

    for outer_idx, (inner, outer_test) in enumerate(random_split_outer.split(idx, y)):
        outer_test_idx[outer_idx] = outer_test
        outer_train_val_idx[outer_idx] = inner
        for inner_idx, (train_val, inner_test) in enumerate(kf_split_inner.split(inner, y[inner])):
            inner_test_idx[outer_idx, inner_idx, :inner_test.shape[0]] = inner[inner_test]

            (train, val) = next(random_split_inner.split(inner[train_val], y[inner[train_val]]))
            inner_train_idx[outer_idx, inner_idx, :train.shape[0]] = inner[train_val][train]
            inner_val_idx[outer_idx, inner_idx, :val.shape[0]] = inner[train_val][val]

    for test_idx in range(config['number_train_test_splits']):
        train_val = outer_train_val_idx[test_idx]
        np.random.shuffle(train_val)  # ensure that the outer cv split is different to the inner cv split
        for cv_idx, (train, val) in enumerate(kf_split_outer.split(train_val, y[train_val])):
            outer_train_idx[test_idx, cv_idx, :train.shape[0]] = train_val[train]
            outer_val_idx[test_idx, cv_idx, :val.shape[0]] = train_val[val]

    return outer_test_idx, inner_test_idx, inner_train_idx, inner_val_idx, outer_val_idx, outer_train_idx


if __name__ == "__main__":
    # KIRC DATASPLIT
    config = {'number_train_test_splits': 10,  # number of random test train splits (MCCV)
              'cv_fold_size': 5,  # number of folds (k in k-fold)
              'cv_fold_size_es_outer': 5,
              'split_test_fraction': 0.2,  # fraction of test set in MCCV outer split
              'number_samples': 506,
              'split_early_stopping_fraction': 0.2,
              'data_file': './data/kirc_random_nodes_preprocessed_all.pkl'}

    outer_test_idx, inner_test_idx, inner_train_idx, inner_val_idx, outer_val_idx, outer_train_idx = split(config)
    print(outer_test_idx, inner_test_idx, inner_train_idx, inner_val_idx, outer_val_idx, outer_train_idx)
    print(outer_test_idx.shape)

    '''
    torch.save({'outer_test_idx': outer_test_idx,
                'inner_test_idx': inner_test_idx,
                'inner_train_idx': inner_train_idx,
                'inner_val_idx': inner_val_idx,
                'outer_val_idx': outer_val_idx,
                'outer_train_idx': outer_train_idx},
               './data/data_splits_KIRC.pt')
    '''

    # BRCA DATASPLIT
    config = {'number_train_test_splits': 10,  # number of random test train splits (MCCV)
              'cv_fold_size': 5,  # number of folds (k in k-fold)
              'cv_fold_size_es_outer': 5,
              'split_test_fraction': 0.2,  # fraction of test set in MCCV outer split
              'number_samples': 689,
              'split_early_stopping_fraction': 0.2,
              'data_file': './data/brca_graphs_preprocessed_all.pkl'}

    outer_test_idx, inner_test_idx, inner_train_idx, inner_val_idx, outer_val_idx, outer_train_idx = split(config)
    print(outer_test_idx, inner_test_idx, inner_train_idx, inner_val_idx, outer_val_idx, outer_train_idx)
    print(outer_test_idx.shape)

    '''
    torch.save({'outer_test_idx': outer_test_idx,
                'inner_test_idx': inner_test_idx,
                'inner_train_idx': inner_train_idx,
                'inner_val_idx': inner_val_idx,
                'outer_val_idx': outer_val_idx,
                'outer_train_idx': outer_train_idx},
               './data/data_splits_BRCA.pt')
    '''

