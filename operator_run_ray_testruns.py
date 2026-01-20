# Created by julia at 25.07.2024
# Adapted version of operator_run.py for pretests without complex data splitting scheme

import importlib

import ray.tune
from lightning.pytorch.loggers import CometLogger

import lightning as pl
import os
import sys
import numpy as np
import torch

from gnn_models import MLP, GCN, GAT, GCN2MLP, GAT2MLP, ChebNet, ChebNet2MLP
from data import KIRCDataModule
import utils
import callbacks
from process_results import process_predictions, analyze_results
from data_splitting import split
import argparse

from lightning.pytorch.callbacks import ModelCheckpoint

from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
from ray.train import RunConfig, ScalingConfig, FailureConfig
from ray.train.torch import TorchTrainer
from ray import tune, train
from ray.air import session

parser = argparse.ArgumentParser(prog='operator_run_ray')
parser.add_argument('config_path', type=str, help="Config file used for the requested run.")
args = parser.parse_args()

module_name = os.path.basename(args.config_path).replace('.py', '')
spec = importlib.util.spec_from_file_location(module_name, args.config_path)
if spec is None:
    raise ImportError(f"Config file/module {args.config_path} not found!")
config = importlib.util.module_from_spec(spec)
#sys.modules[config] = config
spec.loader.exec_module(config)
print(f'----- USED CONFIG FILE: {args.config_path}')


def main():
    np.random.seed(10)

    # Dataframe directory and file for store csv with results
    results_directory = os.path.join(config.config['result_dir'], config.config['project_name'],
                                     config.config['experiment_name'])
    comet_directory = os.path.join(config.config['comet_dir'], config.config['project_name'],
                                   config.config['experiment_name'])
    results_outer_file = os.path.join(results_directory, 'results_outer.csv')
    torch_trainer_file_outer = os.path.join(results_directory, 'TorchTrainerOuter')

    # Create result and comet logger directory
    if not os.path.exists(results_directory):
        os.makedirs(results_directory, exist_ok=True)
    if not os.path.exists(os.path.join(results_directory, 'inner_results')):
        os.makedirs(os.path.join(results_directory, 'inner_results'), exist_ok=True)
    if not os.path.exists(os.path.join(results_directory, 'outer_results')):
        os.makedirs(os.path.join(results_directory, 'outer_results'), exist_ok=True)

    if not os.path.exists(comet_directory):
        os.makedirs(comet_directory, exist_ok=True)

    # Warn if tuner already finished for this experiment (outer, final result file only stored if finished!)
    if os.path.exists(results_outer_file):
        raise ValueError('Tuner for this experiment is already finished and end results are stored. Either remove the '
                         'experiment result directory or change the name of your experiment and/or project!')

    # Set hyperparameter search space and trials for inner run (for outer: only best from inner run)
    search_space_outer = utils.search_space_to_tune_grid(config.search_space)
    num_samples = 1  # runs of ray tune per hyperparameter grid point

    # Create and store indices for inner and outer cross-validation,
    # then utilize datasplit numbers (inner & outer) as hyperparameters in  ray tune
    outer_test_idx, inner_test_idx, inner_train_idx, inner_val_idx, outer_val_idx, outer_train_idx = split(config.config, seed=42)

    # Configurate used computational resources
    if 'resources_per_worker' and 'num_workers' in config.config:
        scaling_config = ScalingConfig(
            use_gpu=True, resources_per_worker=config.config['resources_per_worker'],
            num_workers=config.config['num_workers']
        )

    else: #default, for KIRC pretest configs where not needed (not contained)
        scaling_config = ScalingConfig(
            use_gpu=True, resources_per_worker={"CPU": 31, "GPU": 1},
        )

    ray_trainer_outer = TorchTrainer(
        tune.with_parameters(train_func_inner, train_idx=outer_train_idx, val_idx=outer_val_idx,
                             test_idx=outer_test_idx),
        scaling_config=scaling_config
    )

    # Automated resume, if previous tuning not finished
    if tune.Tuner.can_restore(torch_trainer_file_outer):
        print('------RESUMING OUTER TUNER')
        tuner_outer = tune.Tuner.restore(
            torch_trainer_file_outer,
            trainable=ray_trainer_outer,
            resume_errored=True,
            resume_unfinished=True
        )
    else:
        print('------NEW OUTER TUNER')
        tuner_outer = tune.Tuner(
            ray_trainer_outer,
            param_space={"train_loop_config": search_space_outer},
            tune_config=tune.TuneConfig(
                # metric="val_accuracy",
                # mode="max",
                num_samples=num_samples,
            ),
            run_config=RunConfig(
                storage_path=results_directory,
                name='TorchTrainerOuter',
                failure_config=FailureConfig(max_failures=10))
        )

    results = tuner_outer.fit()

    # Store results of inner folds in dataframe
    df_outer = results.get_dataframe()
    df_outer.to_csv(results_outer_file, index=False, sep=',')
    outer_results_df = process_predictions(df_outer, results_directory, results_outer_file, 'inner')
    analyze_results(outer_results_df, results_directory, 'inner')

    return 0


def train_func_inner(hyperparameters, train_idx=None, val_idx=None, test_idx=None):
    # Seed for reproducibility
    pl.seed_everything(42, workers=True)

    # Name variables
    model_name = utils.resolve_model(config.config['model_name'])
    experiment_name = config.config['experiment_name']
    project_name = config.config['project_name']

    # Callbacks
    early_stopping = callbacks.create_early_stopping_callback(patience=config.config['patience'])
    if 'dropout_rate' in hyperparameters:
        run_id = 'lr{0},wd{1},dropout{2},hidden_layers{3},width{4},batch_size{5},test_round{6},fold{7}_inner'.format(
            str(hyperparameters['learning_rate']), str(hyperparameters['weight_decay']),
            str(hyperparameters['dropout_rate']), str(config.config['number_hidden_layers']),
            str(config.config['width']), str(hyperparameters['batch_size']), str(hyperparameters['dataset_test_round']),
            str(hyperparameters['dataset_fold_round']))
    else:
        run_id = 'lr{0},wd{1},dropout_normal{2},dropout_att{3},hidden_layers{4},width{5},batch_size{6},test_round{7},fold{8}_inner'.format(
            str(hyperparameters['learning_rate']), str(hyperparameters['weight_decay']),
            str(hyperparameters['dropout_rate_normal']), str(hyperparameters['dropout_rate_attention']),
            str(config.config['number_hidden_layers']), str(config.config['width']),
            str(hyperparameters['batch_size']), str(hyperparameters['dataset_test_round']),
            str(hyperparameters['dataset_fold_round']))

    checkpoint_callback = callbacks.create_checkpoint_callback(run_id, project_name, experiment_name,
                                                               config.config['checkpoint_dir'])

    # Logger
    if config.config['comet_logger']:
        if config.config['offline']:
            comet_logger = CometLogger(save_dir=os.path.join(config.config['comet_dir'], project_name,experiment_name, 'inner'),
                                       project_name=project_name, experiment_name=experiment_name+'_inner',
                                       offline=True, log_graph=True)
        else:
            comet_logger = CometLogger(save_dir=os.path.join(config.config['comet_dir'], project_name,experiment_name, 'inner'),
                                       project_name=project_name, experiment_name=experiment_name+'_inner',
                                       api_key=os.environ.get("COMET_API_KEY").replace("'", ""))

    else:
        comet_logger = None

    # Select and clear up data indices
    test_idx = test_idx[hyperparameters['dataset_test_round']]
    test_idx = test_idx[test_idx >= 0]
    train_idx = train_idx[hyperparameters['dataset_test_round'], hyperparameters['dataset_fold_round']]
    train_idx = train_idx[train_idx >= 0]  # cut -1 values(used for matching max shape)
    val_idx = val_idx[hyperparameters['dataset_test_round'], hyperparameters['dataset_fold_round']]
    val_idx = val_idx[val_idx >= 0]

    # Data
    #With Coarsening Index given
    if 'coarsening' in config.config and config.config['coarsening']:
        datamodule = KIRCDataModule(
            config.config['data_file'], batch_size=hyperparameters['batch_size'],
            num_workers=config.config['num_workers_datamodule'], train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            coarsening_dir=config.config['coarsening_dir']
        )
    #Without Coarsening Index
    else:
        datamodule = KIRCDataModule(
            config.config['data_file'], batch_size=hyperparameters['batch_size'],
            num_workers=config.config['num_workers_datamodule'], train_idx=train_idx, val_idx=val_idx,test_idx=test_idx
        )


    # Model
    if config.config['model_name'] == 'MLP':
        model = MLP(
            number_input_features=config.config['number_input_features'],
            number_hidden_layers=config.config['number_hidden_layers'],
            width=config.config['width'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate=hyperparameters['dropout_rate'],
            residual=config.config['residual'],
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    elif config.config['model_name'] == 'GCN':
        model = GCN(
            number_input_channels=config.config['number_input_channels'],
            number_hidden_layers=config.config['number_hidden_layers'],
            width=config.config['width'],
            output_dim=config.config['output_dim'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate=hyperparameters['dropout_rate'],
            residual=config.config['residual'],
            coarsening=config.config['coarsening'] if 'coarsening' in config.config else False, #coarsening key not in all configs
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    elif config.config['model_name'] == 'GAT':
        model = GAT(
            number_input_channels=config.config['number_input_channels'],
            number_hidden_layers=config.config['number_hidden_layers'],
            heads=config.config['heads'],
            width=config.config['width'],
            output_dim=config.config['output_dim'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate_normal=hyperparameters['dropout_rate_normal'],
            dropout_rate_attention=hyperparameters['dropout_rate_attention'],
            residual=config.config['residual'],
            coarsening=config.config['coarsening'] if 'coarsening' in config.config else False, # coarsening key not in all configs
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    elif config.config['model_name'] == 'GCN2MLP':
        model = GCN2MLP(
            number_input_channels=config.config['number_input_channels'],
            number_output_nodes=config.config['number_nodes'],
            number_hidden_layers=config.config['number_hidden_layers'],
            width=config.config['width'],
            output_dim=config.config['output_dim'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate=hyperparameters['dropout_rate'],
            residual=config.config['residual'],
            coarsening=config.config['coarsening'] if 'coarsening' in config.config else False, # coarsening key not in all configs
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    elif config.config['model_name'] == 'GAT2MLP':
        model = GAT2MLP(
            number_input_channels=config.config['number_input_channels'],
            number_output_nodes=config.config['number_nodes'],
            number_hidden_layers=config.config['number_hidden_layers'],
            heads=config.config['heads'],
            width=config.config['width'],
            output_dim=config.config['output_dim'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate_normal=hyperparameters['dropout_rate_normal'],
            dropout_rate_attention=hyperparameters['dropout_rate_attention'],
            residual=config.config['residual'],
            coarsening=config.config['coarsening'] if 'coarsening' in config.config else False, # coarsening key not in all configs
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    elif config.config['model_name'] == 'ChebNet':
        model = ChebNet(
            number_input_channels=config.config['number_input_channels'],
            number_hidden_layers=config.config['number_hidden_layers'],
            width=config.config['width'],
            output_dim=config.config['output_dim'],
            K=config.config['K'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate=hyperparameters['dropout_rate'],
            residual=config.config['residual'],
            coarsening=config.config['coarsening'] if 'coarsening' in config.config else False, # coarsening key not in all configs
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    elif config.config['model_name'] == 'ChebNet2MLP':
        model = ChebNet2MLP(
            number_input_channels=config.config['number_input_channels'],
            number_output_nodes=config.config['number_nodes'],
            number_hidden_layers=config.config['number_hidden_layers'],
            width=config.config['width'],
            output_dim=config.config['output_dim'],
            K=config.config['K'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            dropout_rate=hyperparameters['dropout_rate'],
            residual=config.config['residual'],
            coarsening=config.config['coarsening'] if 'coarsening' in config.config else False, # coarsening key not in all configs
            dense=config.config['dense'],
            width_hidden_reduced_dense=config.config['width_hidden_reduced_dense']
        )

    else:
        raise ValueError('Model name invalid! No model with this name available! Please choose one of: MLP, GCN, GAT.')

    # Trainer
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=config.config['num_epochs'],
        callbacks=[checkpoint_callback, early_stopping],
        plugins=[RayLightningEnvironment()],
        logger=comet_logger,
        devices='auto',
        accelerator='auto',
        deterministic=True,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        strategy=RayDDPStrategy(),
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model=model, datamodule=datamodule)

    number_epochs = (trainer.current_epoch - 1) - early_stopping.wait_count

    # Load best model and activate prediction storage
    best_model = model_name.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.store_predict = True

    # Test and validation
    val = trainer.validate(best_model, datamodule)
    test = trainer.test(best_model, datamodule)
    # Rename result of inner split!
    val[0]['inner_es_loss'] = val[0].pop('val_loss')
    val[0]['inner_es_accuracy'] = val[0].pop('val_accuracy')
    test[0]['val_loss'] = test[0].pop('test_loss')
    test[0]['val_accuracy'] = test[0].pop('test_accuracy')

    # Metrics results report
    val_test_results = {**val[0], **test[0], 'number_epochs': number_epochs}
    train.report(val_test_results)

    # Predictions storage (with trial id in name for identification)
    val_targets, val_outputs, val_predictions = best_model.val_targets,best_model.val_outputs, best_model.val_predictions
    test_targets, test_outputs, test_predictions = best_model.test_targets, best_model.test_outputs, best_model.test_predictions

    trial_id = session.get_trial_id()
    prediction_directory = os.path.join(config.config['result_dir'], config.config['project_name'],
                                        config.config['experiment_name'], 'inner_results', trial_id)

    torch.save({'test_targets': test_targets, 'test_outputs': test_outputs, 'test_predictions': test_predictions,
                'val_targets': val_targets, 'val_outputs': val_outputs,'val_predictions': val_predictions},
                prediction_directory + '.pt')


if __name__ == "__main__":
    sys.exit(main())
