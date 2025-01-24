import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import load_data as ld
import gdp_model as dm
import train_model as mod
import gdp_prediction as pred
import tensorflow as tf
import os


## intersect genes among omics, select 10000 genes with highest var
# def gene_filter(dataframes):
#     mRNA_genes = set(dataframes['mRNATPM_map'].columns)
#     cnv_genes = set(dataframes['cnv_map'].columns)
#     overlapping_genes = list(mRNA_genes.intersection(cnv_genes))
#
#     mRNA_var = dataframes['mRNATPM_map'][overlapping_genes].std()
#     cnv_var = dataframes['cnv_map'][overlapping_genes].std()
#
#     top_mRNA_genes = mRNA_var.nlargest(min(10, len(mRNA_var))).index.tolist() ## change later
#     top_cnv_genes = cnv_var.nlargest(min(10, len(cnv_var))).index.tolist()
#     top_genes = list(set(top_mRNA_genes) & set(top_cnv_genes))
#
#     # top_genes = overlapping_genes   ## keep all the genes
#
#     # all_genes = list(overlapping_genes)
#     return overlapping_genes, top_genes


### filter function
# def impute_df(dataframes, all_genes, top_genes):
def impute_df(dataframes):

    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        if df.isna().any().any():
            df = df.loc[:, df.isnull().mean() < 0.8]
            df = df.fillna(df.mean())

        if df.max().max() > 100:
            df = np.log2(df + 1)

        # df = df.iloc[:, 1:100]   ## delete later

        # unique_genes = set(df.columns) - set(all_genes)
        # selected_genes = list(set(top_genes).union(unique_genes))
        # selected_genes = list(set(selected_genes).intersection(df.columns))
        # df = df[selected_genes]
        processed_dataframes[name] = df

    return processed_dataframes


## process train dataframes
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    processed_train_dataframes['survival'] = train_dataframes['survival']

    for name, df in train_dataframes.items():
        if name == 'survival':
            continue

        # Z-score transformation for tables different from survival
        scaler = StandardScaler()
        z_scored_data = scaler.fit_transform(df)  ## standardScaler automatically converts 0 sd to 1 sd.
        df_new = pd.DataFrame(z_scored_data, columns=df.columns, index=df.index)
        scalers[name] = scaler

        if name == 'clinical':
            for col in df_new.columns:
                if df[col].nunique() <= 5:
                    df_new[col] = df[col]

        processed_train_dataframes[name] = df_new
    return (processed_train_dataframes, scalers)


# Now process test dataframes
def process_testdf(processed_train_dataframes, test_dataframes, scalers):
    processed_test_dataframes = {}
    processed_test_dataframes['survival'] = test_dataframes['survival']

    for name, df in test_dataframes.items():
        if name == 'survival':
            continue

        # Z-score transformation for tables different from survival
        # Use the scaler from training data if available
        z_scored_data = scalers[name].transform(df)
        df_new = pd.DataFrame(z_scored_data, columns=df.columns, index=df.index)

        if name == 'clinical':
            for col in df_new.columns:
                if df[col].nunique() <= 5:
                    df_new[col] = df[col]

        processed_test_dataframes[name] = df_new
    return processed_test_dataframes


def assign_column_groups(mRNA, cnv, clinical):
    mRNA_genes = set(mRNA.columns)
    cnv_genes = set(cnv.columns)

    overlapping_genes = mRNA_genes.intersection(cnv_genes)
    unique_mRNA_genes = mRNA_genes - cnv_genes
    unique_cnv_genes = cnv_genes - mRNA_genes
    big_df = pd.concat([mRNA, cnv, clinical], axis=1)

    group_mapping = {}
    current_group = 1

    for gene in overlapping_genes:
        group_mapping[gene] = current_group
        current_group += 1

    for gene in unique_mRNA_genes:
        group_mapping[gene] = current_group
        current_group += 1

    for gene in unique_cnv_genes:
        group_mapping[gene] = current_group
        current_group += 1

    clinical_group = current_group
    for col in clinical.columns:
        group_mapping[col] = clinical_group

    group_ids = [group_mapping.get(col, 0) for col in big_df.columns]

    return group_ids


def find_divisors(number):
    divisors = []
    for i in range(1, int(number ** 0.5) + 1):
        if number % i == 0:
            divisors.append(i)
            if i != number // i:
                divisors.append(number // i)
    return sorted(divisors)


def find_larger_numbers(numbers, threshold):
    return [num for num in numbers if num >= threshold]


### functions for training and prediction
save_folder = 'SavedModel'
saver_file_prefix = 'Saved'
if os.path.exists("./" + save_folder) == False:
    os.makedirs("./" + save_folder)

## hyperparameters
model = 'NN'  # can set to 'linear' as well
hidden_nodes = [200, 100]
activation = 'relu'  # or set to 'sigmoid' or 'tanh'
dropout_keep_rate = 0.8
alpha = 0.9
scale = 0.5
delta = 0.01
reg_type = "group_lasso"  # available: 'lasso', 'l2', 'group_lasso', 'sparse_group_lasso', 'none'
initial_learning_rate = 0.0001
max_steps = 100

def fit(TrainDataset, batch_size, dataset, current_time, fold):
    if os.path.exists(os.path.join("./", save_folder, dataset)) == False:
        os.makedirs(os.path.join("./", save_folder, dataset))
    if os.path.exists(os.path.join("./", save_folder, dataset, 'Time' + str(current_time))) == False:
        os.makedirs(os.path.join("./", save_folder, dataset, 'Time' + str(current_time)))

    feature_groups = TrainDataset.train.feature_groups
    feature_size = TrainDataset.train.feature_size
    dm.FEATURE_SIZE = feature_size

    with tf.Graph().as_default():
        feature_pl, at_risk_pl, date_pl, censor_pl = mod.placeholder_inputs(batch_size)
        isTrain_pl = tf.placeholder(tf.bool, shape=())
        if model == 'NN':
            inf_output_pl = dm.inference(feature_pl, hidden_nodes, activation, dropout_keep_rate, isTrain_pl)
        elif model == 'linear':
            inf_output_pl = dm.inference_linear(feature_pl)

        loss = dm.loss(inf_output_pl, censor_pl, feature_groups, batch_size, alpha, scale, delta, reg_type)
        train_op = dm.training(loss, initial_learning_rate)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)

        for step in range(max_steps):
            print('Training step: ', step)
            feed_dict = mod.fill_feed_dict(TrainDataset.train.next_batch(batch_size), feature_pl, at_risk_pl, date_pl,
                                           censor_pl)
            feed_dict[isTrain_pl] = True
            _, loss_value, output = sess.run([train_op, loss, inf_output_pl], feed_dict=feed_dict)

            if (step + 1) % 5 == 0:
                print('Step: %d' % (step + 1))
            if (step + 1) == max_steps:
                checkpoint_file = os.path.join("./", save_folder, dataset, 'Time' + str(current_time), 'Fold' + str(fold) + saver_file_prefix)
                saver.save(sess, checkpoint_file, global_step=step)


def predict(dict, group, dataset, current_time, fold):
    X = pd.concat([dict['mRNATPM_map'], dict['cnv_map'], dict['clinical']], axis=1).to_numpy('float32')
    T = np.ones(X.shape[0])
    O = np.ones(X.shape[0])

    ValDataset = ld.DataSet(X, T, O, group)
    sample_size = ValDataset.patients_num
    features = np.asarray(list(ValDataset.features)).astype('float32')

    with tf.Graph().as_default():
        feature_pl = pred.placeholder_inputs(sample_size)
        isTrain_pl = tf.placeholder(tf.bool,
                                    shape=())  # boolean value to check if it is during training optimization process
        if model == 'NN':
            inf_output_pl = dm.inference(feature_pl, hidden_nodes, activation, 1, isTrain_pl)
        elif model == 'linear':
            inf_output_pl = dm.inference_linear(feature_pl)

        hazard_pl = dm.hazard(inf_output_pl)

        # init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # sess = tf.Session()
        checkpoint_file = os.path.join("./", save_folder, dataset, 'Time' + str(current_time), 'Fold' + str(fold) +
                                       saver_file_prefix + '-' + str(max_steps - 1))

        with tf.Session() as sess:
            # sess.run(init)
            saver.restore(sess, checkpoint_file)
            # Predict hazard
            feed_dict = {feature_pl: features, isTrain_pl: False}
            hazard = sess.run(hazard_pl, feed_dict=feed_dict)
            # hazard = hazard.flatten()
            h_saved = pd.DataFrame(hazard)

    return h_saved


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')