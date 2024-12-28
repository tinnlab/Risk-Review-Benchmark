from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

## remove columns with NA/ zero std for each data type
def filter_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        # if name == 'clinical':
        #     colkeeps = ["age_at_initial_pathologic_diagnosis", "history_other_malignancy__no"]
        #     df = df.loc[:, colkeeps]

        # Remove columns with any NA values
        df = df.dropna(axis=1)
        if df.max().max() > 100:
            df = np.log2(df + 1)

        # Remove columns with zero standard deviation for each data type
        cols_to_keep = [col for col in df.columns if df[col].std() > 0]
        df = df[cols_to_keep]
        processed_dataframes[name] = df

    return processed_dataframes

## process train data
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    # processed_train_dataframes['survival'] = train_dataframes['survival']
    # processed_train_dataframes['clinical'] = train_dataframes['clinical']
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
    # processed_test_dataframes['survival'] = test_dataframes['survival']
    # processed_test_dataframes['clinical'] = test_dataframes['clinical']

    for name, df in test_dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        # Use the scaler from training data if available
        if name in scalers:
            tf_data = scalers[name].transform(df)
            df = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
        processed_test_dataframes[name] = df
    return processed_test_dataframes