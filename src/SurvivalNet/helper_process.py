import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import random
import Bayesian_Optimization as BayesOpt
from survivalnet.train import train
import theano


## knn imputation
def knn_impute(X, k=1):
    X = np.array(X, dtype=float)
    X_imputed = X.copy()
    na_columns = np.where(np.any(np.isnan(X), axis=0))[0]
    for col in na_columns:
        missing_idx = np.isnan(X[:, col])
        non_missing_idx = ~missing_idx

        if not np.any(missing_idx):
            continue
        if not np.any(non_missing_idx):
            raise ValueError("Cannot impute column with all missing values")

        for idx in np.where(missing_idx)[0]:
            row = X[idx]
            valid_features1 = ~np.isnan(row)
            valid_rows = np.where(non_missing_idx)[0]
            valid_features2 = ~np.any(np.isnan(X[valid_rows]), axis=0)
            valid_features = valid_features1 & valid_features2

            distances = cdist(row[np.newaxis, valid_features], X[valid_rows][:, valid_features])[0]
            k_nearest = valid_rows[np.argsort(distances)[:k]]
            X_imputed[idx, col] = np.mean(X[k_nearest, col])

    return X_imputed


## impute function
def impute_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        # variances = df.var(skipna=True)  # Calculate variance of each column
        # top_indices = variances.nlargest(
        #     min(50, df.shape[1])).index  # Get indices of top 50 variances
        # df = df[top_indices]

        if df.isnull().any().any():
            df = df.dropna(axis=1, how='all')  ## drop columns with all na values
            # KNN Imputation with k=1
            df_imputed = knn_impute(df)
            df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        if df.max().max(skipna=True) > 100:
            df = np.log2(df + 1)

        processed_dataframes[name] = df
    return processed_dataframes


def manual_kfold_split(y, n_splits=10, random_seed=1234):
    y = np.array(y)
    alive_indices = np.where(y == 1)[0]
    dead_indices = np.where(y == 0)[0]

    np.random.seed(random_seed)
    np.random.shuffle(alive_indices)
    np.random.shuffle(dead_indices)

    n_alive_per_fold = len(alive_indices) // n_splits
    n_dead_per_fold = len(dead_indices) // n_splits

    fold_indices = []

    for i in range(n_splits):
        if i < n_splits - 1:
            val_alive = alive_indices[i * n_alive_per_fold: (i + 1) * n_alive_per_fold]
            val_dead = dead_indices[i * n_dead_per_fold: (i + 1) * n_dead_per_fold]
        else:
            val_alive = alive_indices[i * n_alive_per_fold:]
            val_dead = dead_indices[i * n_dead_per_fold:]

        val_indices = np.concatenate([val_alive, val_dead])
        train_indices = np.array([idx for idx in range(len(y)) if idx not in val_indices])
        fold_indices.append((train_indices, val_indices))

    return fold_indices


## split patients into train, and val sets
def train_val_split(alive_list, dead_list, random_seed, val_size=0.2):
    random.seed(random_seed)
    n_alive_val = int(len(alive_list) * val_size)
    n_dead_val = int(len(dead_list) * val_size)

    alive_val_indices = set(random.sample(range(len(alive_list)), n_alive_val))
    dead_val_indices = set(random.sample(range(len(dead_list)), n_dead_val))

    alive_train = [x for i, x in enumerate(alive_list) if i not in alive_val_indices]
    alive_val = [alive_list[i] for i in alive_val_indices]

    dead_train = [x for i, x in enumerate(dead_list) if i not in dead_val_indices]
    dead_val = [dead_list[i] for i in dead_val_indices]

    return alive_train, alive_val, dead_train, dead_val


## process train data
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    processed_train_dataframes['survival'] = train_dataframes['survival']
    cate_cols = []

    for name, df in train_dataframes.items():
        if name == 'survival':
            continue

        variance = df.var()
        non_zero_var_cols = variance[variance > 0].index
        df = df[non_zero_var_cols]

        mean = df.mean()
        std = df.std()
        scalers[name] = {'mean': mean, 'std': std, 'columns': non_zero_var_cols}

        df_new = (df - mean) / std

        if name == 'clinical':
            cate_cols = []
            for col in df.columns:
                if df[col].nunique() <= 5 and df[col].max() <= 5:
                    cate_cols.append(col)
            df_new[cate_cols] = df[cate_cols]
        processed_train_dataframes[name] = df_new
    return processed_train_dataframes, scalers, cate_cols


## process val/test data
def process_val_test_df(dataframes, scalers, cate_cols):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']

    for name, df in dataframes.items():
        if name == 'survival':
            continue

        # Get stored statistics
        mean = scalers[name]['mean']
        std = scalers[name]['std']
        cols = scalers[name]['columns']

        # Keep only columns that were in training set
        df = df[cols]

        # Apply z-score using training statistics
        df_new = (df - mean) / std

        if name == 'clinical':
            df_new[cate_cols] = df[cate_cols]
        processed_dataframes[name] = df_new

    return processed_dataframes


## tranform dict of dataframes into the right input
def transform_df(dataframes):
    new_dict = {}
    mrna_df = dataframes['mRNATPM']
    cnv_df = dataframes['cnv']
    clin_df = dataframes['clinical']
    surv_df = dataframes['survival']

    mrna_df = mrna_df.add_prefix('mRNA_')
    cnv_df = cnv_df.add_prefix('CNV_')

    merged_df = clin_df.merge(mrna_df, left_index=True, right_index=True) \
        .merge(cnv_df, left_index=True, right_index=True)

    input_array = merged_df.values.astype(np.float32)
    status_array = surv_df['status'].values.astype(np.int32)
    time_array = surv_df['time'].values.astype(np.float32)

    new_dict['X'] = input_array
    new_dict['T'] = time_array
    new_dict['O'] = status_array

    return new_dict


## transform data for prediction
def transform_df_pred(dataframes):
    mrna_df = dataframes['mRNATPM']
    cnv_df = dataframes['cnv']
    clin_df = dataframes['clinical']

    mrna_df = mrna_df.add_prefix('mRNA_')
    cnv_df = cnv_df.add_prefix('CNV_')

    merged_df = clin_df.merge(mrna_df, left_index=True, right_index=True) \
        .merge(cnv_df, left_index=True, right_index=True)

    input_array = merged_df.values.astype(np.float32)
    return input_array


## function
def fit(pretrain_set, dataset, current_time, fold, train_set, val_set, epochs=40, pretrain_config=None, opt='GDLS', do_bayes_opt=True):
    if do_bayes_opt == True:
        print '***Model Selection with BayesOpt***'
        _, bo_params = BayesOpt.tune(dataset, current_time, fold)
        n_layers = int(bo_params[0])
        n_hidden = int(bo_params[1])
        do_rate = bo_params[2]
        nonlin = theano.tensor.nnet.relu if bo_params[3] > .5 else np.tanh
        lambda1 = bo_params[4]
        lambda2 = bo_params[5]
    else:
        n_layers = 1
        n_hidden = 100
        do_rate = 0.5
        lambda1 = 0
        lambda2 = 0
        nonlin = np.tanh  # or nonlin = theano.tensor.nnet.relu

    finetune_config = {'ft_lr': 0.0001, 'ft_epochs': epochs}

    print '*** Model Assesment ***'
    _, train_cindices, _, val_cindices, _, _, model, _ = train(pretrain_set,
                                                                train_set, val_set, pretrain_config,
                                                                finetune_config, n_layers,
                                                                n_hidden, dropout_rate=do_rate, lambda1=lambda1,
                                                                lambda2=lambda2,
                                                                non_lin=nonlin, optim=opt, verbose=True,
                                                                earlystp=False)
    return model, n_hidden, n_layers


## define the predict function
def predict(model, new_data, n_hidden, n_layers):
    test_masks = [np.ones((new_data.shape[0], n_hidden), dtype='int64') for i in range(n_layers)]

    test, _ = model.build_finetune_functions(learning_rate=0.0)

    observed = np.ones(new_data.shape[0], dtype='int32')
    at_risk = np.ones(new_data.shape[0], dtype='int32')

    _, risk_scores, _ = test(new_data, observed, at_risk, 0, *test_masks)
    return risk_scores
