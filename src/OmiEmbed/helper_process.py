import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from datasets import create_single_dataloader
from params.train_params import TrainParams
from util import util
from models import create_model
from util.visualizer import Visualizer


YChr_Genes = pd.read_csv("../../AllData/YChr_Genes.csv", header=0, sep=",")
Remove_MethProbes = pd.read_csv("../../AllData/Remove_MethProbes.csv", header=0, sep=",")


## set up some parameters
param = TrainParams().parse()
if param.deterministic:
    util.setup_seed(param.seed)


## function to filter and impute data
def filim_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        if name == "mRNATPM":
            drop_columns = list(set(YChr_Genes['YChr_GeneSymbol']) & set(df.columns))
            df = df.drop(drop_columns, axis=1)

        if name == "meth450":
            drop_columns = list(set(Remove_MethProbes['ProbeID']) & set(df.columns))
            df = df.drop(drop_columns, axis=1)

        if df.isna().any().any():
            df = df.loc[:, df.isnull().mean() <= 0.1]
            df = df.fillna(df.mean())

        if df.max(skipna=True).max() > 100:
            df = np.log2(df + 1)
        processed_dataframes[name] = df
    return processed_dataframes


## process train dataframes
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    processed_train_dataframes['survival'] = train_dataframes['survival']
    processed_train_dataframes['clinical'] = train_dataframes['clinical']
    processed_train_dataframes['meth450'] = train_dataframes['meth450']

    for name, df in train_dataframes.items():
        if name in ['survival', 'clinical', 'meth450']:
            continue

        # minmax normalization for tables different from survival, clinical
        scaler = MinMaxScaler()
        tf_data = scaler.fit_transform(df)
        df = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
        scalers[name] = scaler
        processed_train_dataframes[name] = df
    return (processed_train_dataframes, scalers)


## Now process test dataframes
def process_testdf(test_dataframes, scalers):
    processed_test_dataframes = {}
    processed_test_dataframes['survival'] = test_dataframes['survival']
    processed_test_dataframes['clinical'] = test_dataframes['clinical']
    processed_test_dataframes['meth450'] = test_dataframes['meth450']

    for name, df in test_dataframes.items():
        if name in ['survival', 'clinical', 'meth450']:
            continue

        # Use the scaler from training data if available
        if name in scalers:
            tf_data = scalers[name].transform(df)
            df = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
        processed_test_dataframes[name] = df
    return processed_test_dataframes



## save dataframes to data folder
def save_df(dataframes):
    mRNA = dataframes['mRNATPM']
    meth = dataframes['meth450']
    miRNA = dataframes['miRNA']
    survival = dataframes['survival']
    clinical = dataframes['clinical']
    reg_df = clinical[['age_at_initial_pathologic_diagnosis']]
    class_df = clinical[['history_other_malignancy__no']]
    mRNA.T.to_csv("./TCGA-data/A.tsv", sep='\t', header=True)
    meth.T.to_csv("./TCGA-data/B.tsv", sep='\t', header=True)
    miRNA.T.to_csv("./TCGA-data/C.tsv", sep='\t', header=True)
    survival.to_csv("./TCGA-data/survival.tsv", sep='\t', header=True)
    reg_df.to_csv("./TCGA-data/values.tsv", sep='\t', header=True)
    class_df.to_csv("./TCGA-data/labels.tsv", sep='\t', header=True)


## train function
def train():
    # Dataset related
    dataloader, sample_list = create_single_dataloader(param, enable_drop_last=True)
    print('The size of training set is {}'.format(len(dataloader)))
    # Get the dimension of input omics data
    param.omics_dims = dataloader.get_omics_dims()
    if param.downstream_task in ['classification', 'multitask', 'alltask']:
        # Get the number of classes for the classification task
        if param.class_num == 0:
            param.class_num = dataloader.get_class_num()
        if param.downstream_task != 'alltask':
            print('The number of classes: {}'.format(param.class_num))
    if param.downstream_task in ['regression', 'multitask', 'alltask']:
        # Get the range of the target values
        values_min = dataloader.get_values_min()
        values_max = dataloader.get_values_max()
        if param.regression_scale == 1:
            param.regression_scale = values_max
        print('The range of the target values is [{}, {}]'.format(values_min, values_max))
    if param.downstream_task in ['survival', 'multitask', 'alltask']:
        # Get the range of T
        survival_T_min = dataloader.get_survival_T_min()
        survival_T_max = dataloader.get_survival_T_max()
        if param.survival_T_max == -1:
            param.survival_T_max = survival_T_max
        print('The range of survival T is [{}, {}]'.format(survival_T_min, survival_T_max))

    # Model related
    model = create_model(param)  # Create a model given param.model and other parameters
    model.setup(param)  # Regular setup for the model: load and print networks, create schedulers
    visualizer = Visualizer(param)  # Create a visualizer to print results

    # Start the epoch loop
    visualizer.print_phase(model.phase)
    for epoch in range(param.epoch_count, param.epoch_num + 1):  # outer loop for different epochs
        epoch_start_time = time.time()  # Start time of this epoch
        model.epoch = epoch
        # TRAINING
        model.set_train()  # Set train mode for training
        iter_load_start_time = time.time()  # Start time of data loading for this iteration
        output_dict, losses_dict, metrics_dict = model.init_log_dict()  # Initialize the log dictionaries
        if epoch == param.epoch_num_p1 + 1:
            model.phase = 'p2'  # Change to supervised phase
            visualizer.print_phase(model.phase)
        if epoch == param.epoch_num_p1 + param.epoch_num_p2 + 1:
            model.phase = 'p3'  # Change to supervised phase
            visualizer.print_phase(model.phase)
        if param.save_latent_space and epoch == param.epoch_num:
            latent_dict = model.init_latent_dict()

        # Start training loop
        for i, data in enumerate(dataloader):  # Inner loop for different iteration within one epoch
            model.iter = i
            dataset_size = len(dataloader)
            actual_batch_size = len(data['index'])
            iter_start_time = time.time()  # Timer for computation per iteration
            if i % param.print_freq == 0:
                load_time = iter_start_time - iter_load_start_time  # Data loading time for this iteration
            model.set_input(data)  # Unpack input data from the output dictionary of the dataloader
            model.update()  # Calculate losses, gradients and update network parameters
            model.update_log_dict(output_dict, losses_dict, metrics_dict,
                                  actual_batch_size)  # Update the log dictionaries
            if param.save_latent_space and epoch == param.epoch_num:
                latent_dict = model.update_latent_dict(latent_dict)  # Update the latent space array
            if i % param.print_freq == 0:  # Print training losses and save logging information to the disk
                comp_time = time.time() - iter_start_time  # Computational time for this iteration
                visualizer.print_train_log(epoch, i, losses_dict, metrics_dict, load_time, comp_time, param.batch_size,
                                           dataset_size)
            iter_load_start_time = time.time()

        # Model saving
        if param.save_model:
            if param.save_epoch_freq == -1:  # Only save networks during last epoch
                if epoch == param.epoch_num:
                    print('Saving the model at the end of epoch {:d}'.format(epoch))
                    # model.save_networks(str(epoch))
                    model.save_networks('latest')
            elif epoch % param.save_epoch_freq == 0:  # Save both the generator and the discriminator every <save_epoch_freq> epochs
                print('Saving the model at the end of epoch {:d}'.format(epoch))
                model.save_networks('latest')
                # model.save_networks(str(epoch))

        train_time = time.time() - epoch_start_time
        current_lr = model.update_learning_rate()  # update learning rates at the end of each epoch
        visualizer.print_train_summary(epoch, losses_dict, output_dict, train_time, current_lr)

        if param.save_latent_space and epoch == param.epoch_num:
            visualizer.save_latent_space(latent_dict, sample_list)
    print('Training is done!')


## merge dataframes
def merge_df(train_dataframes, test_data_frames):
    cb_df = {}
    for name, df in train_dataframes.items():
        cb_df[name] = pd.concat([df, test_data_frames[name]], axis=0)
    return cb_df


## predict function
def predict():
    # Dataset related
    dataloader, sample_list = create_single_dataloader(param, shuffle=False)  # No shuffle for testing
    print('The size of testing set is {}'.format(len(dataloader)))
    # Get sample list for the dataset
    param.sample_list = dataloader.get_sample_list()
    # Get the dimension of input omics data
    param.omics_dims = dataloader.get_omics_dims()
    if param.downstream_task == 'classification' or param.downstream_task == 'multitask':
        # Get the number of classes for the classification task
        if param.class_num == 0:
            param.class_num = dataloader.get_class_num()
        print('The number of classes: {}'.format(param.class_num))
    if param.downstream_task == 'regression' or param.downstream_task == 'multitask':
        # Get the range of the target values
        values_min = dataloader.get_values_min()
        values_max = dataloader.get_values_max()
        if param.regression_scale == 1:
            param.regression_scale = values_max
        print('The range of the target values is [{}, {}]'.format(values_min, values_max))
    if param.downstream_task == 'survival' or param.downstream_task == 'multitask':
        # Get the range of T
        survival_T_min = dataloader.get_survival_T_min()
        survival_T_max = dataloader.get_survival_T_max()
        if param.survival_T_max == -1:
            param.survival_T_max = survival_T_max
        print('The range of survival T is [{}, {}]'.format(survival_T_min, survival_T_max))

    # Model related
    model = create_model(param)  # Create a model given param.model and other parameters
    model.setup(param)  # Regular setup for the model: load and print networks, create schedulers
    visualizer = Visualizer(param)  # Create a visualizer to print results

    # TESTING
    model.set_eval()
    test_start_time = time.time()  # Start time of testing
    output_dict, losses_dict, metrics_dict = model.init_log_dict()  # Initialize the log dictionaries
    if param.save_latent_space:
        latent_dict = model.init_latent_dict()

    # Start testing loop
    for i, data in enumerate(dataloader):
        dataset_size = len(dataloader)
        actual_batch_size = len(data['index'])
        model.set_input(data)  # Unpack input data from the output dictionary of the dataloader
        model.test()  # Run forward to get the output tensors
        model.update_log_dict(output_dict, losses_dict, metrics_dict, actual_batch_size)  # Update the log dictionaries
        if param.save_latent_space:
            latent_dict = model.update_latent_dict(latent_dict)  # Update the latent space array
        # if i % param.print_freq == 0:  # Print testing log
        #     visualizer.print_test_log(param.epoch_to_load, i, losses_dict, metrics_dict, param.batch_size, dataset_size)

    test_time = time.time() - test_start_time
    # visualizer.print_test_summary(param.epoch_to_load, losses_dict, output_dict, test_time)
    visualizer.save_output_dict(output_dict)
    if param.save_latent_space:
        visualizer.save_latent_space(latent_dict, sample_list)
    print('Prediction is done!')


## clear the data folder
def clear_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.unlink(file_path)
