# Created by julia at 29.07.2024

# Model which should be tested (incl. dense/res) and basic configurations fixed over the experiments
config = {
    # Epochs and early stopping
    'num_epochs': 10000,
    'patience': 100,

    # Logger
    'comet_logger': True,
    'offline': True,

    # Home directory instead of scratch (scratch not available on a100!)
    'data_dir': "/user/{username}/GNN4PPI/data",
    'checkpoint_dir': "/user/{username}/GNN4PPI/checkpoints",
    'result_dir': "/user/{username}/GNN4PPI/results",
    'comet_dir': "/user/{username}/GNN4PPI/comet_logs",
    'num_workers_datamodule': 1,
    'data_split_file': "data_splits_KIRC.pt",

    # Datasplit (MCCV, k-fold CV)
    'number_train_test_splits': 10, #2, # number of random test train splits (MCCV)
    'cv_fold_size': 5, # number of folds (k in k-fold), inner
    'split_test_fraction': 0.2, # fraction of test set in MCCV outer split, outer
    'split_early_stopping_fraction': 0.2, # for inner early stopping random splits, inner
    'cv_fold_size_es_outer':5 # for outer CV early stopping, outer

}

# Hyperparameters examined in grid search
search_space= {
    'learning_rate':[0.1, 0.01, 0.001, 0.0001, 0.00001],
    'weight_decay': [0, 0.0001, 0.001, 0.01, 0.1],
    'dropout_rate': [0, 0.2, 0.4, 0.6, 0.8],  # 0 = no dropout, dropout_rate = prob. for removing neuron
    'batch_size': [506],

    # For dataset iterations (cross validation, inner + outer)
    'dataset_test_round': [0], #list(range(config['number_train_test_splits'])),
    'dataset_fold_round': [0] #list(range(config['cv_fold_size']))

}
