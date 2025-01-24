import os
from functools import reduce
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

## Please update the below paths if necessary
datPath = '../../AllData/ReviewPaper_Data_5kfeats'
processedPath = '../../AllData/TF-ProcessData_cv2'

if os.path.exists(processedPath) == False:
    os.makedirs(processedPath)

alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]

for dataset in alldatasets:
    print(dataset)

    if os.path.exists(processedPath + '/' + dataset) == False:
        os.makedirs(processedPath + '/' + dataset)

    dataTypes = ["mRNATPM_map", "cnv_map", "miRNA", "meth450_map", "clinical", "survival"]
    dataTypesUsed = ["mRNATPM_map", "cnv_map", "meth450_map", "clinical", "survival"]

    dataList = {}
    for dataType in dataTypes:
        df = pd.read_csv(datPath + "/" + dataset + "/" + dataType + ".csv", header=0, index_col=0)
        dataList[dataType] = df

    common_rows = reduce(
        lambda x, y: x.intersection(y),
        [df.index for df in dataList.values()]
    )

    # To get DataFrames with only common rows
    common_dataList = {name: df.loc[common_rows] for name, df in dataList.items()}
    common_dataList = {name: df for name, df in common_dataList.items() if name in dataTypesUsed}
    shared_columns = list(set(common_dataList['mRNATPM_map'].columns) & set(common_dataList['cnv_map'].columns) & set(common_dataList['meth450_map'].columns))

    def process_df(dataframes, keepcols):
        processed_dataframes = {}
        processed_dataframes['survival'] = dataframes['survival']

        for name, df in dataframes.items():
            if name == 'survival':
                continue

            if name == 'clinical':
                df['age_at_initial_pathologic_diagnosis'] = (df['age_at_initial_pathologic_diagnosis'] >= df['age_at_initial_pathologic_diagnosis'].median()).astype(int)
                processed_dataframes[name] = df
                continue

            df = df[keepcols]
            if df.max().max() > 100:
                df = np.log2(df + 1)
            df = df.fillna(0)
            processed_dataframes[name] = df
        return processed_dataframes

    common_dataList = process_df(common_dataList, shared_columns)

    ### split data in to train and test set
    survival_df = common_dataList['survival']

    for current_time in range(1, 6):
        print(f'Running Time: {current_time}')

        if os.path.exists(os.path.join(processedPath, dataset, 'Time' + str(current_time))) == False:
            os.makedirs(os.path.join(processedPath, dataset, 'Time' + str(current_time)))
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
            print(f'Running Fold: {fold}')
            train_dataframes = {name: df.iloc[train_idx]
                                for name, df in common_dataList.items()}

            test_dataframes = {name: df.iloc[val_idx]
                               for name, df in common_dataList.items()}

            train_dataframes['mRNATPM_map'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/mRNA_train_' + str(fold) + '.csv', sep=',', header=True)
            train_dataframes['cnv_map'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/cnv_train_' + str(fold) + '.csv', sep=',', header=True)
            train_dataframes['meth450_map'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/meth_train_' + str(fold) + '.csv', sep=',', header=True)
            train_dataframes['clinical'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/clinical_train_' + str(fold) + '.csv', sep=',', header=True)
            train_dataframes['survival'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/survival_train_' + str(fold) + '.csv', sep=',', header=True)

            test_dataframes['mRNATPM_map'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/mRNA_val_' + str(fold) + '.csv', sep=',', header=True)
            test_dataframes['cnv_map'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/cnv_val_' + str(fold) + '.csv', sep=',', header=True)
            test_dataframes['meth450_map'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/meth_val_' + str(fold) + '.csv', sep=',', header=True)
            test_dataframes['clinical'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/clinical_val_' + str(fold) + '.csv', sep=',', header=True)
            test_dataframes['survival'].to_csv(processedPath + '/' + dataset + '/Time' + str(current_time) + '/survival_val_' + str(fold) + '.csv', sep=',', header=True)

    print('done!')



