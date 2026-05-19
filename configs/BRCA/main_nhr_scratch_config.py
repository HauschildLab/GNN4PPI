# Created by julia at 22.12.2025

# Model which should be tested (incl. dense/res) and basic configurations fixed over the experiments
config = {
    # Epochs and early stopping
    'num_epochs': 10000,
    'patience': 100,

    # Logger
    'comet_logger': True,
    'offline': True,

    # Home directory instead of scratch (scratch not available on a100!)
    'data_dir': "/mnt/lustre-grete/usr/{username}/gnn_for_ppi/data",
    'checkpoint_dir': "/mnt/lustre-grete/usr/{username}/gnn_for_ppi/new_results/BRCA/checkpoints",
    'result_dir': "/mnt/lustre-grete/usr/{username}/gnn_for_ppi/new_results/BRCA/results",
    'comet_dir': "/mnt/lustre-grete/usr/{username}/gnn_for_ppi/new_results/BRCA/comet_logs",
    'data_split_file': "data_splits_BRCA.pt",

    ################
    'num_workers_datamodule': 1,

    # Datasplit (MCCV, k-fold CV)
    'number_train_test_splits': 10, #2, # number of random test train splits (MCCV)
    'cv_fold_size': 5, # number of folds (k in k-fold), inner
    'split_test_fraction': 0.2, # fraction of test set in MCCV outer split, outer
    'split_early_stopping_fraction': 0.2, # for inner early stopping random splits, inner
    'cv_fold_size_es_outer': 5 # for outer CV early stopping, outer

}

# Hyperparameters examined in grid search
search_space = {
    'batch_size': [689],

    # For dataset iterations (cross validation, inner + outer)
    'dataset_test_round': list(range(config['number_train_test_splits'])),
    'dataset_fold_round': list(range(config['cv_fold_size']))
}
