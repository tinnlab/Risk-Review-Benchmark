import sys
import shutil
sys.path.append("./")

import os
import time
import warnings
from functools import reduce
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
from helper_process import *

# torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'
datPath = '../../AllData/ReviewPaper_Data_5kfeats'
resPath = '../../run-results'
timerecPath = "../../time-rec"

dataset = param.dataset # This gets the dataset argument
current_time = int(param.cvtime)
fold_assign = int(param.FoldID)

if __name__ == "__main__":
    if os.path.exists(resPath + "/OmiEmbed") == False:
        os.makedirs(resPath + "/OmiEmbed")
    if os.path.exists(timerecPath + "/OmiEmbed") == False:
        os.makedirs(timerecPath + "/OmiEmbed")

    if os.path.exists("./TCGA-data") == False:
        os.makedirs("./TCGA-data")

    warnings.filterwarnings('ignore')
    # Get parameters
    print("****** dataset_time_fold: " + dataset + "_" + str(current_time) + "_" + str(fold_assign) + " ******")
    if os.path.exists(resPath + "/OmiEmbed/" + dataset) == False:
        os.makedirs(resPath + "/OmiEmbed/" + dataset)
    if os.path.exists(timerecPath + "/OmiEmbed/" + dataset) == False:
        os.makedirs(timerecPath + "/OmiEmbed/" + dataset)

    dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
    dataTypesUsed = ["mRNATPM", "miRNA", "meth450", "clinical", "survival"]

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
    common_dataList = filim_df(common_dataList)

    survival_df = common_dataList['survival']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)

    if os.path.exists(os.path.join(resPath, "OmiEmbed", dataset, 'Time' + str(current_time))) == False:
        os.makedirs(os.path.join(resPath, "OmiEmbed", dataset, 'Time' + str(current_time)))
    if os.path.exists(os.path.join(timerecPath, "OmiEmbed", dataset, 'Time' + str(current_time))) == False:
        os.makedirs(os.path.join(timerecPath, "OmiEmbed", dataset, 'Time' + str(current_time)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
        if fold != fold_assign:
            continue
        print(fold)

        train_dataframes = {name: df.iloc[train_idx]
                            for name, df in common_dataList.items()}

        test_dataframes = {name: df.iloc[val_idx]
                           for name, df in common_dataList.items()}

        train_dataframes, scalers = process_traindf(train_dataframes)
        test_dataframes = process_testdf(test_dataframes, scalers)

        survTrain = train_dataframes['survival']
        survTest = test_dataframes['survival']
        save_df(train_dataframes)

        train_patients = survTrain.index
        test_patients = survTest.index

        start_time = time.time()

        try:
            train()

            ## prediction
            cb_df = merge_df(train_dataframes, test_dataframes)
            save_df(cb_df)
            surv_random = cb_df['survival'].copy()
            surv_random['status'] = 1
            np.random.seed(1234)  # Set random seed
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

        end_time = time.time()
        record_time = end_time - start_time
        print(f'Running Time: {record_time:.2f} seconds')

        time_df = pd.DataFrame({
            'dataset': [dataset],
            'time_point': [current_time],
            'fold': [fold],
            'runtime_seconds': [record_time]
        })

        predTrain = pd.concat([predTrain, survTrain], axis=1)
        predVal = pd.concat([predVal, survTest], axis=1)

        predTrain.to_csv(resPath + "/OmiEmbed" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv', sep=',',
                         header=True)
        predVal.to_csv(resPath + "/OmiEmbed" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv', sep=',',
                       header=True)
        time_df.to_csv(timerecPath + "/OmiEmbed" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv', header=True)

        clear_files("./TCGA-data")
        shutil.rmtree("./checkpoints")
