# Created by julia at 19.09.2024
from configs_pretest.KIRC import main_config
import os

config = main_config.config
search_space = main_config.search_space

# Model and Experiment
config['model_name'] = 'MLP'
config['experiment_name'] = 'FinalTest_HPC_full_corr'
config['project_name'] = 'Test_DenseMLP_v4_methyl'

# Residual / Dense
config['residual'] = False
config['dense'] = True
config['width_hidden_reduced_dense'] = True

# Data (full/expr/methyl)
config['data_mode'] = 'methyl'


# Set model parameters (evtl. more than one for tuning-> search space instead of config)
config['number_hidden_layers'] = 3
config['width'] = 64


### DO NOT CHANGE!!! ###
# Dataset
config['number_samples'] = 506
config['number_nodes'] = 1594

if config['data_mode']=='full':
    config['number_input_channels'] = 2
    config['data_file'] = os.path.join(config['data_dir'], 'kirc_random_nodes_preprocessed_all.pkl')

elif config['data_mode']=='expr':
    config['number_input_channels'] = 1
    config['data_file'] = os.path.join(config['data_dir'], 'kirc_random_nodes_preprocessed_expr.pkl')

elif config['data_mode'] == 'methyl':
    config['number_input_channels'] = 1
    config['data_file'] = os.path.join(config['data_dir'], 'kirc_random_nodes_preprocessed_methyl.pkl')

else:
    raise ValueError('Data_mode in config file invalid! Choose between "full", "expr" and "methyl".')

config['number_input_features'] = config['number_nodes']*config['number_input_channels']
