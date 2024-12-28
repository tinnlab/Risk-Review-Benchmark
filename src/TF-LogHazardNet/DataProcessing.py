import os
from functools import reduce
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
processedPath = '/nfs/blanche/share/daotran/SurvivalPrediction/TF-ProcessData'

if os.path.exists(processedPath) == False:
    os.makedirs(processedPath)

all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]

for fold in all_folds:
    print(fold)

    if os.path.exists(processedPath + '/' + fold) == False:
        os.makedirs(processedPath + '/' + fold)

    dataTypes = ["mRNATPM_map", "cnv_map", "miRNA", "meth450_map", "clinical", "survival"]
    dataTypesUsed = ["mRNATPM_map", "cnv_map", "meth450_map", "clinical", "survival"]

    dataList = {}
    for dataType in dataTypes:
        df = pd.read_csv(datPath + "/" + fold + "/" + dataType + ".csv")
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
    alive_patients = survival_df[survival_df['status'] == 0].index.tolist()
    dead_patients = survival_df[survival_df['status'] == 1].index.tolist()

    for seed in range(1, 11):
        print("****** seed " + str(seed) + " ******")
        alive_train, alive_test = train_test_split(alive_patients, test_size=0.2, random_state=seed)
        dead_train, dead_test = train_test_split(dead_patients, test_size=0.2, random_state=seed)
        train_patients = alive_train + dead_train
        test_patients = alive_test + dead_test

        train_dataframes = {name: df.loc[train_patients]
                            for name, df in common_dataList.items()}

        test_dataframes = {name: df.loc[test_patients]
                           for name, df in common_dataList.items()}

        train_dataframes['mRNATPM_map'].to_csv(processedPath + '/' + fold + '/mRNA_train_' + str(seed) + '.csv', sep=',', header=True)
        train_dataframes['cnv_map'].to_csv(processedPath + '/' + fold + '/cnv_train_' + str(seed) + '.csv', sep=',', header=True)
        train_dataframes['meth450_map'].to_csv(processedPath + '/' + fold + '/meth_train_' + str(seed) + '.csv', sep=',', header=True)
        train_dataframes['clinical'].to_csv(processedPath + '/' + fold + '/clinical_train_' + str(seed) + '.csv', sep=',', header=True)
        train_dataframes['survival'].to_csv(processedPath + '/' + fold + '/survival_train_' + str(seed) + '.csv', sep=',', header=True)

        test_dataframes['mRNATPM_map'].to_csv(processedPath + '/' + fold + '/mRNA_val_' + str(seed) + '.csv', sep=',', header=True)
        test_dataframes['cnv_map'].to_csv(processedPath + '/' + fold + '/cnv_val_' + str(seed) + '.csv', sep=',', header=True)
        test_dataframes['meth450_map'].to_csv(processedPath + '/' + fold + '/meth_val_' + str(seed) + '.csv', sep=',', header=True)
        test_dataframes['clinical'].to_csv(processedPath + '/' + fold + '/clinical_val_' + str(seed) + '.csv', sep=',', header=True)
        test_dataframes['survival'].to_csv(processedPath + '/' + fold + '/survival_val_' + str(seed) + '.csv', sep=',', header=True)

    print('done!')



