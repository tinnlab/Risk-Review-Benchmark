from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch


## replace NAs with 0
def filter_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        df = df.fillna(0)
        processed_dataframes[name] = df
    return processed_dataframes


## process train data
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    processed_train_dataframes['survival'] = train_dataframes['survival']

    for name, df in train_dataframes.items():
        if name == 'survival':
            continue

        if name == 'clinical':
            scaler = MinMaxScaler()
            tf_data = scaler.fit_transform(df)
            df_new = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
            scalers[name] = scaler
            for col in df_new.columns:
                if df[col].nunique() <= 5 and df[col].max() <= 5:
                    df_new[col] = df[col]
            processed_train_dataframes[name] = df_new
            continue

        if name == 'mRNATPM':
            df = df.loc[:, df.var() > 7]

        if name == 'CNV':
            df = df.loc[:, df.var() > 0.2]

        if df.max().max() > 100:
            df = np.log2(df + 1)

        scaler = MinMaxScaler()
        tf_data = scaler.fit_transform(df)
        df_new = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
        scalers[name] = scaler
        processed_train_dataframes[name] = df_new
    return (processed_train_dataframes, scalers)


## Now process test dataframes
def process_testdf(processed_train_dataframes, test_dataframes, scalers):
    processed_test_dataframes = {}
    processed_test_dataframes['survival'] = test_dataframes['survival']

    for name, df in test_dataframes.items():
        if name == 'survival':
            continue

        if name == 'clinical':
            tf_data = scalers[name].transform(df)
            df_new = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
            for col in df_new.columns:
                if df[col].nunique() <= 5 and df[col].max() <= 5:
                    df_new[col] = df[col]
            processed_test_dataframes[name] = df_new
            continue

        df = df[processed_train_dataframes[name].columns]
        if df.max().max() > 100:
            df = np.log2(df + 1)
        tf_data = scalers[name].transform(df)
        df_new = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
        processed_test_dataframes[name] = df_new
    return processed_test_dataframes


## generate modality_dim dict
def mod_dim(train_dataframes):
    res_dict = {}
    res_dict['clinical'] = train_dataframes['clinical'].shape[1]
    res_dict['mRNA'] = train_dataframes['mRNATPM'].shape[1]
    res_dict['miRNA'] = train_dataframes['miRNA'].shape[1]
    res_dict['CNV'] = train_dataframes['cnv'].shape[1]
    return res_dict


## get the categorical columns, embedding size for categorical features and number of numeric columns
def find_cate(clin_df, colKeep):
    clin_df_new = clin_df[colKeep]
    cate_cols = []
    for col in clin_df_new.columns:
        if clin_df_new[col].nunique() <= 5 and clin_df_new[col].max() <= 5:
            cate_cols.append(col)
    clin_df_new_cate = clin_df_new[cate_cols]
    embedding_sizes = []
    for col in clin_df_new_cate.columns:
        n_categories = clin_df_new_cate[col].max() + 1
        # n_categories = n_categories + 1
        if n_categories == 1:
            n_categories += 1
        embedding_dim = round(n_categories / 2)
        embedding_sizes.append((int(n_categories), int(embedding_dim)))
    n_continuous = len(colKeep) - len(cate_cols)
    return cate_cols, embedding_sizes, n_continuous


## generate data for prediction
def gen_preddat(test_dataframes, train_dataframes, cate_cols):
    pred_dict = {}
    clin_train = train_dataframes['clinical']
    for name, df in test_dataframes.items():
        if name == 'survival':
            continue

        if name == 'clinical':
            nume_cols = list(set(clin_train.columns) - set(cate_cols))
            clin_data_categorical = df[cate_cols]
            clin_data_continuous = df[nume_cols]
            clin_cat = clin_data_categorical.values.tolist()
            clin_cont = clin_data_continuous.values.tolist()
            clin_cate = np.array(clin_cat).astype(np.int64)
            clin_cate = torch.from_numpy(clin_cate)
            pred_dict['clinical_categorical'] = clin_cate

            clin_conti = np.array(clin_cont).astype(np.float32)
            clin_conti = torch.from_numpy(clin_conti)
            pred_dict['clinical_continuous'] = clin_conti
            continue

        x_train_all = torch.tensor(df.values, dtype=torch.float32)
        if name == 'mRNATPM':
            pred_dict['mRNA'] = x_train_all

        if name == 'cnv':
            pred_dict['CNV'] = x_train_all

        if name == 'miRNA':
            pred_dict['miRNA'] = x_train_all
    return pred_dict





