import sys
import shutil
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/OmiEmbed")

import os
import argparse
import warnings
from functools import reduce
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from helper_process import *

# torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'
datPath = '../../AllData/ReviewPaper_Data'
resPath = '../../run-results'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--FoldID', type=str, required=True)
args = parser.parse_args()

# Get parameters
fold = args.dataset  # This gets the dataset argument
seed = int(args.FoldID)

if __name__ == "__main__":
    if os.path.exists(resPath + "/OmiEmbed") == False:
        os.makedirs(resPath + "/OmiEmbed")

    if os.path.exists("./TCGA-data") == False:
        os.makedirs("./TCGA-data")

    warnings.filterwarnings('ignore')
    # Get parameters
    print("****** dataset_seed: " + fold + "_" + str(seed) + " ******")
    if os.path.exists(resPath + "/OmiEmbed/" + fold) == False:
        os.makedirs(resPath + "/OmiEmbed/" + fold)

    dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
    dataTypesUsed = ["mRNATPM", "miRNA", "meth450", "clinical", "survival"]

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
    common_dataList = filim_df(common_dataList)

    survival_df = common_dataList['survival']
    alive_patients = survival_df[survival_df['status'] == 0].index.tolist()
    dead_patients = survival_df[survival_df['status'] == 1].index.tolist()

    alive_train, alive_test = train_test_split(alive_patients, test_size=0.2, random_state=seed)
    dead_train, dead_test = train_test_split(dead_patients, test_size=0.2, random_state=seed)
    train_patients = alive_train + dead_train
    test_patients = alive_test + dead_test

    train_dataframes = {name: df.loc[train_patients]
                        for name, df in common_dataList.items()}

    test_dataframes = {name: df.loc[test_patients]
                       for name, df in common_dataList.items()}

    train_dataframes, scalers = process_traindf(train_dataframes)
    test_dataframes = process_testdf(test_dataframes, scalers)

    survTrain = train_dataframes['survival']
    survTest = test_dataframes['survival']
    save_df(train_dataframes)

    try:
        train()

        ## prediction
        cb_df = merge_df(train_dataframes, test_dataframes)
        save_df(cb_df)
        surv_random = cb_df['survival'].copy()
        surv_random['status'] = 1
        np.random.seed(seed)  # Set random seed
        surv_random['time'] = np.random.randint(1, 1001, size=len(surv_random))
        surv_random.to_csv("./TCGA-data/survival.tsv", sep='\t', header=True)
        predict()

        predAll = pd.read_csv("./checkpoints/test/down_output/survival_function.tsv",
                              sep="\t", header=0, index_col=0)
        predTrain = predAll.loc[train_patients,]
        predVal = predAll.loc[test_patients,]

        unique_time = [round(float(col)) for col in predAll.columns]
        column_names = [f"predTrain_{time}" for time in unique_time]
        predTrain.columns = column_names

        column_names = [f"predVal_{time}" for time in unique_time]
        predVal.columns = column_names
    except:
        time_list = list(range(0, survTrain['time'].max() + 10, 100))
        column_names = [f"predTrain_{time}" for time in time_list]
        predTrain = pd.DataFrame(np.nan,
                              index=range(len(survTrain)),
                              columns=column_names)

        time_list = list(range(0, survTest['time'].max() + 10, 100))
        column_names = [f"predVal_{time}" for time in time_list]
        predVal = pd.DataFrame(np.nan,
                                 index=range(len(survTest)),
                                 columns=column_names)

        predTrain.index = survTrain.index
        predVal.index = survTest.index

    predTrain = pd.concat([predTrain, survTrain], axis=1)
    predVal = pd.concat([predVal, survTest], axis=1)

    predTrain.to_csv(resPath + "/OmiEmbed" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                     header=True)
    predVal.to_csv(resPath + "/OmiEmbed" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                   header=True)

    clear_files("./TCGA-data")
    shutil.rmtree("./checkpoints")
