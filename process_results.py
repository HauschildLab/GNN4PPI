# Created by julia at 12.09.2024
import os

import torch
import torchmetrics
import pandas as pd
import argparse

accuracy = torchmetrics.classification.BinaryAccuracy()
precision = torchmetrics.classification.BinaryPrecision()
recall = torchmetrics.classification.BinaryRecall()
specificity = torchmetrics.classification.BinarySpecificity()
confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()
f1_score = torchmetrics.classification.BinaryF1Score()
mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
auroc = torchmetrics.classification.BinaryAUROC()
auprc = torchmetrics.classification.BinaryAveragePrecision()


def process_predictions(hyperparameters_df=None, results_dir=None, file_name_csv=None, mode=None):
    # Check if use via command-line (no arguments given) and not as function
    if hyperparameters_df is None or results_dir is None or file_name_csv is None:
        request = 'command_line'

        parser = argparse.ArgumentParser(prog='process_results')
        parser.add_argument('results_dir_name')
        args = parser.parse_args()
        results_dir = args.results_dir_name

        # Check if extended results file already exists
        for file in os.listdir(results_dir):
            if file.endswith("_extended.csv"):
                raise ValueError(
                    "Extended results file already exists! Change the result directory or check the results "
                    "file in the given directory!")

        # Get information about hyperparameters for trials (normally only one result csv per directory)
        for file in os.listdir(results_dir):
            if file.endswith(".csv"):
                file_name_csv = os.path.join(results_dir, file)
                hyperparameters_df = pd.read_csv(file_name_csv)

        mode = 'inner'

    else:
        request = 'function'

    if mode == 'inner':
        val = 'inner_es'
        test = 'val'

    elif mode == 'outer':
        val = 'outer_es'
        test = 'test'
        # Rename column names for hyperparameters (incl. overwrite existing file)
        hyperparameters_df['config/train_loop_config/dataset_test_round'] = hyperparameters_df[
            'config/train_loop_config/test_params/config/train_loop_config/dataset_test_round']
        hyperparameters_df['config/train_loop_config/learning_rate'] = hyperparameters_df[
            'config/train_loop_config/test_params/config/train_loop_config/learning_rate']
        hyperparameters_df['config/train_loop_config/weight_decay'] = hyperparameters_df[
            'config/train_loop_config/test_params/config/train_loop_config/weight_decay']
        hyperparameters_df['config/train_loop_config/batch_size'] = hyperparameters_df[
            'config/train_loop_config/test_params/config/train_loop_config/batch_size']

        # Check if GAT or not -> dropout rate or dr_normal and  _attention processed
        if 'config/train_loop_config/test_params/config/train_loop_config/dropout_rate_normal' in hyperparameters_df.columns:
            hyperparameters_df['config/train_loop_config/dropout_rate_normal'] = hyperparameters_df[
                'config/train_loop_config/test_params/config/train_loop_config/dropout_rate_normal']
            hyperparameters_df['config/train_loop_config/dropout_rate_attention'] = hyperparameters_df[
                'config/train_loop_config/test_params/config/train_loop_config/dropout_rate_attention']
            hyperparameters_df.drop(
                columns=['config/train_loop_config/test_params/config/train_loop_config/dataset_test_round',
                         'config/train_loop_config/test_params/config/train_loop_config/learning_rate',
                         'config/train_loop_config/test_params/config/train_loop_config/dropout_rate_normal',
                         'config/train_loop_config/test_params/config/train_loop_config/dropout_rate_attention',
                         'config/train_loop_config/test_params/config/train_loop_config/weight_decay',
                         'config/train_loop_config/test_params/config/train_loop_config/batch_size'], inplace=True)
        else:
            hyperparameters_df['config/train_loop_config/dropout_rate'] = hyperparameters_df[
                'config/train_loop_config/test_params/config/train_loop_config/dropout_rate']
            hyperparameters_df.drop(
                columns=['config/train_loop_config/test_params/config/train_loop_config/dataset_test_round',
                         'config/train_loop_config/test_params/config/train_loop_config/learning_rate',
                         'config/train_loop_config/test_params/config/train_loop_config/dropout_rate',
                         'config/train_loop_config/test_params/config/train_loop_config/weight_decay',
                         'config/train_loop_config/test_params/config/train_loop_config/batch_size'], inplace=True)
        hyperparameters_df.to_csv(file_name_csv, index=False)

    else:
        raise ValueError('Mode not choosen the right way - only "inner" or "outer" possible!')

    hyperparameters_df[val + '_precision'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_recall'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_specificity'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_f1_score'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_mcc'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_auroc'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_auprc'] = pd.Series(dtype='float32')
    hyperparameters_df[val + '_true_positives'] = pd.Series(dtype='int')
    hyperparameters_df[val + '_false_negatives'] = pd.Series(dtype='int')
    hyperparameters_df[val + '_false_positives'] = pd.Series(dtype='int')
    hyperparameters_df[val + '_true_negatives'] = pd.Series(dtype='int')

    hyperparameters_df[test + '_precision'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_recall'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_specificity'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_f1_score'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_mcc'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_auroc'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_auprc'] = pd.Series(dtype='float32')
    hyperparameters_df[test + '_true_positives'] = pd.Series(dtype='int')
    hyperparameters_df[test + '_false_negatives'] = pd.Series(dtype='int')
    hyperparameters_df[test + '_false_positives'] = pd.Series(dtype='int')
    hyperparameters_df[test + '_true_negatives'] = pd.Series(dtype='int')

    # Get trial pred. and targets and print metrices and hyperparameters
    for file in os.listdir(os.path.join(results_dir, mode + '_results')):
        if file.endswith(".pt"):
            # print(f'TRIAL {file[-8:-3]}')
            file_name_pred = os.path.join(results_dir, mode + '_results', file)
            results_pred = torch.load(file_name_pred)

            # Get predictions and targets
            test_targets = results_pred['test_targets']
            test_predictions = results_pred['test_predictions']
            val_targets = results_pred['val_targets']
            val_predictions = results_pred['val_predictions']

            # Compute metrics for validation and test
            val_accuracy = accuracy(val_predictions, val_targets)
            val_precision = precision(val_predictions, val_targets)
            val_recall = recall(val_predictions, val_targets)
            val_specificity = specificity(val_predictions, val_targets)
            val_confusion_matrix = confusion_matrix(val_predictions, val_targets)
            val_f1_score = f1_score(val_predictions, val_targets)
            val_mcc = mcc(val_predictions, val_targets)
            val_auroc = auroc(val_predictions, val_targets)
            val_auprc = auprc(val_predictions, val_targets.type(torch.int64))

            test_accuracy = accuracy(test_predictions, test_targets)
            test_precision = precision(test_predictions, test_targets)
            test_recall = recall(test_predictions, test_targets)
            test_specificity = specificity(test_predictions, test_targets)
            test_confusion_matrix = confusion_matrix(test_predictions, test_targets)
            test_f1_score = f1_score(test_predictions, test_targets)
            test_mcc = mcc(test_predictions, test_targets)
            test_auroc = auroc(test_predictions, test_targets)
            test_auprc = auprc(test_predictions, test_targets.type(torch.int64))

            # Store results in dataframe (in row for current trial)
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_accuracy'] = val_accuracy.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_precision'] = val_precision.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_recall'] = val_recall.item()
            hyperparameters_df.loc[
                hyperparameters_df.trial_id == file[:-3], val + '_specificity'] = val_specificity.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_f1_score'] = val_f1_score.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_mcc'] = val_mcc.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_auroc'] = val_auroc.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_auprc'] = val_auprc.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_true_positives'] = \
                val_confusion_matrix[0, 0].item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_false_negatives'] = \
                val_confusion_matrix[0, 1].item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_false_positives'] = \
                val_confusion_matrix[1, 0].item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], val + '_true_negatives'] = \
                val_confusion_matrix[1, 1].item()

            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_accuracy'] = test_accuracy.item()
            hyperparameters_df.loc[
                hyperparameters_df.trial_id == file[:-3], test + '_precision'] = test_precision.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_recall'] = test_recall.item()
            hyperparameters_df.loc[
                hyperparameters_df.trial_id == file[:-3], test + '_specificity'] = test_specificity.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_f1_score'] = test_f1_score.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_mcc'] = test_mcc.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_auroc'] = test_auroc.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_auprc'] = test_auprc.item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_true_positives'] = \
                test_confusion_matrix[0, 0].item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_false_negatives'] = \
                test_confusion_matrix[0, 1].item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_false_positives'] = \
                test_confusion_matrix[1, 0].item()
            hyperparameters_df.loc[hyperparameters_df.trial_id == file[:-3], test + '_true_negatives'] = \
                test_confusion_matrix[1, 1].item()

    # Store results in file
    hyperparameters_df.to_csv(file_name_csv[:-4] + '_extended' + file_name_csv[-4:], index=False)

    if request == 'command_line':
        return hyperparameters_df, results_dir
    else:
        return hyperparameters_df


def analyze_results(results_df, results_dir, mode=None):
    if mode == 'inner':
        val = 'inner_es'
        test = 'val'
    else:
        val = 'outer_es'
        test = 'test'

    # if GAT is used model (-> different usage of dropout, two types: normal and attention)
    GAT_used = ('config/train_loop_config/dropout_rate_normal' in results_df.columns)

    results_inner_average = pd.DataFrame()
    results_inner_best = pd.DataFrame()

    max_test_idx = results_df['config/train_loop_config/dataset_test_round'].max()

    # Get all unique hyperparameter combinations (inner:all, outer: best from inner runs)
    if GAT_used:
        unique_hyp_combinations = results_df[['config/train_loop_config/dataset_test_round',
                                              'config/train_loop_config/learning_rate',
                                              'config/train_loop_config/weight_decay',
                                              'config/train_loop_config/batch_size',
                                              'config/train_loop_config/dropout_rate_normal',
                                              'config/train_loop_config/dropout_rate_attention']].drop_duplicates()
    else:
        unique_hyp_combinations = results_df[['config/train_loop_config/dataset_test_round',
                                              'config/train_loop_config/learning_rate',
                                              'config/train_loop_config/weight_decay',
                                              'config/train_loop_config/batch_size',
                                              'config/train_loop_config/dropout_rate']].drop_duplicates()

    for idx, values in unique_hyp_combinations.iterrows():
        test_idx = int(values['config/train_loop_config/dataset_test_round'])
        lr = values['config/train_loop_config/learning_rate']
        wd = values['config/train_loop_config/weight_decay']
        bs = int(values['config/train_loop_config/batch_size'])

        if GAT_used:
            dr_normal = values['config/train_loop_config/dropout_rate_normal']
            dr_att = values['config/train_loop_config/dropout_rate_attention']
            hyperparameters_series = results_df[(results_df['config/train_loop_config/learning_rate'] == lr)
                                                & (results_df[
                                                       'config/train_loop_config/dropout_rate_normal'] == dr_normal)
                                                & (results_df[
                                                       'config/train_loop_config/dropout_rate_attention'] == dr_att)
                                                & (results_df[
                                                       'config/train_loop_config/weight_decay'] == wd)
                                                & (results_df[
                                                       'config/train_loop_config/dataset_test_round'] == test_idx)
                                                & (results_df[
                                                       'config/train_loop_config/batch_size'] == bs)
                                                ]

        else:
            dr = values['config/train_loop_config/dropout_rate']
            hyperparameters_series = results_df[(results_df['config/train_loop_config/learning_rate'] == lr)
                                                & (results_df[
                                                       'config/train_loop_config/dropout_rate'] == dr)
                                                & (results_df[
                                                       'config/train_loop_config/weight_decay'] == wd)
                                                & (results_df[
                                                       'config/train_loop_config/dataset_test_round'] == test_idx)
                                                & (results_df[
                                                       'config/train_loop_config/batch_size'] == bs)
                                                ]

        average_val_loss = hyperparameters_series[val + '_loss'].mean()
        average_val_accuracy = hyperparameters_series[val + '_accuracy'].mean()
        average_val_recall = hyperparameters_series[val + '_recall'].mean()
        average_val_precision = hyperparameters_series[val + '_precision'].mean()
        average_val_specificity = hyperparameters_series[val + '_specificity'].mean()
        average_val_f1_score = hyperparameters_series[val + '_f1_score'].mean()
        average_val_mcc = hyperparameters_series[val + '_mcc'].mean()
        average_val_auroc = hyperparameters_series[val + '_auroc'].mean()
        average_val_auprc = hyperparameters_series[val + '_auprc'].mean()
        average_val_true_positives = hyperparameters_series[val + '_true_positives'].mean()
        average_val_false_negatives = hyperparameters_series[val + '_false_negatives'].mean()
        average_val_false_positives = hyperparameters_series[val + '_false_positives'].mean()
        average_val_true_negatives = hyperparameters_series[val + '_true_negatives'].mean()
        average_test_loss = hyperparameters_series[test + '_loss'].mean()
        average_test_accuracy = hyperparameters_series[test + '_accuracy'].mean()
        average_test_precision = hyperparameters_series[test + '_precision'].mean()
        average_test_recall = hyperparameters_series[test + '_recall'].mean()
        average_test_specificity = hyperparameters_series[test + '_specificity'].mean()
        average_test_f1_score = hyperparameters_series[test + '_f1_score'].mean()
        average_test_mcc = hyperparameters_series[test + '_mcc'].mean()
        average_test_auroc = hyperparameters_series[test + '_auroc'].mean()
        average_test_auprc = hyperparameters_series[test + '_auprc'].mean()
        average_test_true_positives = hyperparameters_series[test + '_true_positives'].mean()
        average_test_false_negatives = hyperparameters_series[test + '_false_negatives'].mean()
        average_test_false_positives = hyperparameters_series[test + '_false_positives'].mean()
        average_test_true_negatives = hyperparameters_series[test + '_true_negatives'].mean()
        if GAT_used:
            new_average_df = pd.DataFrame({'config/train_loop_config/dataset_test_round': [test_idx],
                                           'config/train_loop_config/learning_rate': [lr],
                                           'config/train_loop_config/dropout_rate_normal': [dr_normal],
                                           'config/train_loop_config/dropout_rate_attention': [dr_att],
                                           'config/train_loop_config/weight_decay': [wd],
                                           'config/train_loop_config/batch_size': [bs],
                                           'average_' + val + '_loss': [average_val_loss],
                                           'average_' + val + '_accuracy': [average_val_accuracy],
                                           'average_' + val + '_precision': [average_val_precision],
                                           'average_' + val + '_recall': [average_val_recall],
                                           'average_' + val + '_specificity': [average_val_specificity],
                                           'average_' + val + '_f1_score': [average_val_f1_score],
                                           'average_' + val + '_mcc': [average_val_mcc],
                                           'average_' + val + '_auroc': [average_val_auroc],
                                           'average_' + val + '_auprc': [average_val_auprc],
                                           'average_' + val + '_true_positives': [average_val_true_positives],
                                           'average_' + val + '_false_negatives': [average_val_false_negatives],
                                           'average_' + val + '_false_positives': [average_val_false_positives],
                                           'average_' + val + '_true_negatives': [average_val_true_negatives],
                                           'average_' + test + '_loss': [average_test_loss],
                                           'average_' + test + '_accuracy': [average_test_accuracy],
                                           'average_' + test + '_precision': [average_test_precision],
                                           'average_' + test + '_recall': [average_test_recall],
                                           'average_' + test + '_specificity': [average_test_specificity],
                                           'average_' + test + '_f1_score': [average_test_f1_score],
                                           'average_' + test + '_mcc': [average_test_mcc],
                                           'average_' + test + '_auroc': [average_test_auroc],
                                           'average_' + test + '_auprc': [average_test_auprc],
                                           'average_' + test + '_true_positives': [average_test_true_positives],
                                           'average_' + test + '_false_negatives': [average_test_false_negatives],
                                           'average_' + test + '_false_positives': [average_test_false_positives],
                                           'average_' + test + '_true_negatives': [average_test_true_negatives]})
        else:
            new_average_df = pd.DataFrame({'config/train_loop_config/dataset_test_round': [test_idx],
                                           'config/train_loop_config/learning_rate': [lr],
                                           'config/train_loop_config/dropout_rate': [dr],
                                           'config/train_loop_config/weight_decay': [wd],
                                           'config/train_loop_config/batch_size': [bs],
                                           'average_' + val + '_loss': [average_val_loss],
                                           'average_' + val + '_accuracy': [average_val_accuracy],
                                           'average_' + val + '_precision': [average_val_precision],
                                           'average_' + val + '_recall': [average_val_recall],
                                           'average_' + val + '_specificity': [average_val_specificity],
                                           'average_' + val + '_f1_score': [average_val_f1_score],
                                           'average_' + val + '_mcc': [average_val_mcc],
                                           'average_' + val + '_auroc': [average_val_auroc],
                                           'average_' + val + '_auprc': [average_val_auprc],
                                           'average_' + val + '_true_positives': [average_val_true_positives],
                                           'average_' + val + '_false_negatives': [average_val_false_negatives],
                                           'average_' + val + '_false_positives': [average_val_false_positives],
                                           'average_' + val + '_true_negatives': [average_val_true_negatives],
                                           'average_' + test + '_loss': [average_test_loss],
                                           'average_' + test + '_accuracy': [average_test_accuracy],
                                           'average_' + test + '_precision': [average_test_precision],
                                           'average_' + test + '_recall': [average_test_recall],
                                           'average_' + test + '_specificity': [average_test_specificity],
                                           'average_' + test + '_f1_score': [average_test_f1_score],
                                           'average_' + test + '_mcc': [average_test_mcc],
                                           'average_' + test + '_auroc': [average_test_auroc],
                                           'average_' + test + '_auprc': [average_test_auprc],
                                           'average_' + test + '_true_positives': [average_test_true_positives],
                                           'average_' + test + '_false_negatives': [average_test_false_negatives],
                                           'average_' + test + '_false_positives': [average_test_false_positives],
                                           'average_' + test + '_true_negatives': [average_test_true_negatives]})
        results_inner_average = pd.concat([results_inner_average, new_average_df], ignore_index=True)

    # Compute and store best hyperparameter configuration per test split (over all hyperparameter averages)
    if mode == 'inner':
        for test_idx in range(max_test_idx + 1):
            test_results = results_inner_average[results_inner_average['config/train_loop_config/dataset_test_round']
                                                 == test_idx]
            test_results_max = test_results[
                test_results['average_' + test + '_auroc'] == test_results['average_' + test + '_auroc'].max()]
            if len(test_results_max) > 1:
                print(
                    f'----ATTENTION: MORE THAN ONE OPTIMAL HYPERPARAMETER CONFIGURATION FOUND FOR TEST SPLIT {test_idx}')

            results_inner_best = pd.concat([results_inner_best, test_results_max])

        results_inner_average.to_csv(os.path.join(results_dir, 'results_inner_average_over_folds.csv'), index=False)
        results_inner_best.to_csv(os.path.join(results_dir, 'results_inner_best_over_hyp.csv'), index=False)
        return results_inner_best

    else:
        results_inner_average.to_csv(os.path.join(results_dir, 'results_outer_average_over_folds.csv'), index=False)


if __name__ == "__main__":
    results, res_dir = process_predictions()
    analyze_results(results, res_dir, mode='inner')
