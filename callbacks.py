# Created by julia at 29.07.2024
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def create_checkpoint_callback(run_id, project_name, experiment_name, checkpoint_path):
    return ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", every_n_epochs=1,
                           dirpath=checkpoint_path + '/' + project_name + '/' + experiment_name + '/run_' + str(run_id))


def create_early_stopping_callback(patience):
    return EarlyStopping(monitor="val_loss", patience=patience, mode='min')
