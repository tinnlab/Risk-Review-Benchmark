import pandas as pd
import numpy as np
import random
import coxae
from coxae.model import CoxAutoencoderClustering, ConcreteCoxAutoencoderClustering


## impute function
def impute_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        if df.max().max(skipna=True) > 100:
            df = np.log2(df + 1)
        if df.isna().any().any():
            missing_counts = df.isnull().sum()
            columns_to_drop = missing_counts[missing_counts > 1].index  ## drop columns with more than 1 missing value
            df = df.drop(columns=columns_to_drop)
            df_imputed = df.fillna(0)
            df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        processed_dataframes[name] = df
    return processed_dataframes


## train function
def train(train_dataframes, survdat):
    feature_dfs = train_dataframes.copy()
    X_train = {k: feature_dfs[k].values for k in feature_dfs}

    durations_train = survdat["time"].values
    events_train = survdat["status"].values

    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    model = CoxAutoencoderClustering(
        encoding_feature_selector=coxae.feature_selection.CoxPHFeatureSelector(limit_significant=1000,
                                                                               get_most_significant_combination_time_limit=0))
    model.fit({k: X_train[k] for k in X_train}, durations_train, events_train)
    print("training is done!")
    return(model)


## predict function
def predict(model, dataframes):
    data_list = {k: dataframes[k].values for k in dataframes}
    hazards_train = model.hazard(data_list)
    res = pd.DataFrame(hazards_train)
    return res

