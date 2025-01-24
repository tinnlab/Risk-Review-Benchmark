import sys
sys.path.append("./")

import os
import gc
import time
import numpy as np
import pandas as pd
import cPickle
from functools import reduce
from helper_process import *
from survivalnet.optimization import SurvivalAnalysis

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/SurvivalNet") == False:
#     os.makedirs(resPath + "/SurvivalNet")

def clear_memory():
    """Function to aggressively clear memory"""
    gc.collect()
    time.sleep(20) 


def run(datPath, resPath, timerecPath):

    if os.path.exists("./data") == False:
        os.makedirs("./data")

    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]

    for dataset in alldatasets:
        print(dataset)
        if os.path.exists(resPath + "/SurvivalNet/" + dataset) == False:
            os.makedirs(resPath + "/SurvivalNet/" + dataset)
        if os.path.exists(timerecPath + "/SurvivalNet/" + dataset) == False:
            os.makedirs(timerecPath + "/SurvivalNet/" + dataset)
        if os.path.exists(os.path.join("./data", dataset)) == False:
                os.makedirs(os.path.join("./data", dataset))

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "cnv", "clinical", "survival"]

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
        common_dataList = impute_df(common_dataList)

        survival_df = common_dataList['survival']

        # all_times = {}
        for current_time in range(1, 6):
            print('Running Time: ' + str(current_time))

            if os.path.exists(os.path.join(resPath, "SurvivalNet", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(resPath, "SurvivalNet", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join(timerecPath, "SurvivalNet", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(timerecPath, "SurvivalNet", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join("./data", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join("./data", dataset, 'Time' + str(current_time)))

            # fold_times = []
            fold_indices = manual_kfold_split(survival_df['status'], random_seed=current_time)
            for fold in range(len(fold_indices)):
                # if fold < 8:
                #     continue

                train_idx, val_idx = fold_indices[fold]
                print 'Running Fold: ' + str(fold)

                train_dataframes = {name: df.iloc[train_idx]
                                    for name, df in common_dataList.items()}

                test_dataframes = {name: df.iloc[val_idx]
                                for name, df in common_dataList.items()}

                survTrain = train_dataframes['survival']
                survTest = test_dataframes['survival']

                alive_patients = survTrain[survTrain['status'] == 0].index.tolist()
                dead_patients = survTrain[survTrain['status'] == 1].index.tolist()

                alive_train, alive_val, dead_train, dead_val = train_val_split(alive_patients, dead_patients, 1234)

                train_patients = alive_train + dead_train
                val_patients = alive_val + dead_val

                train_dataframes_small = {name: df.loc[train_patients]
                                        for name, df in train_dataframes.items()}
                val_dataframes = {name: df.loc[val_patients]
                                for name, df in train_dataframes.items()}

                train_dataframes_small, scalers, cate_cols = process_traindf(train_dataframes_small)
                val_dataframes = process_val_test_df(val_dataframes, scalers, cate_cols)
                test_dataframes = process_val_test_df(test_dataframes, scalers, cate_cols)

                train_dict = transform_df(train_dataframes_small)
                val_dict = transform_df(val_dataframes)

                # Uses the entire dataset for pretraining
                pretrain_set = np.vstack((train_dict['X'], val_dict['X']))

                train_set = {}
                val_set = {}

                sa = SurvivalAnalysis()
                train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(
                    train_dict['X'],
                    train_dict['T'],
                    train_dict['O']);
                val_set['X'], val_set['T'], val_set['O'], val_set['A'] = sa.calc_at_risk(
                    val_dict['X'],
                    val_dict['T'],
                    val_dict['O']);

                # Writes data sets for bayesopt cost function's use.
                with file('./data/' + dataset + '/Time' + str(current_time) + '/train_set_' + str(fold), 'wb') as f:
                    cPickle.dump(train_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
                with file('./data/' + dataset + '/Time' + str(current_time) + '/val_set_' + str(fold), 'wb') as f:
                    cPickle.dump(val_set, f, protocol=cPickle.HIGHEST_PROTOCOL)

                start_time = time.time()

                try:
                    model, n_hidden, n_layers = fit(pretrain_set, dataset, current_time, fold, train_set, val_set)

                    predTrain = predict(model, pretrain_set, n_hidden, n_layers)
                    predTrain = np.exp(predTrain)
                    predTrain = pd.DataFrame(predTrain, columns=['predTrain'])

                    test_array = transform_df_pred(test_dataframes)
                    predTest = predict(model, test_array, n_hidden, n_layers)
                    predTest = np.exp(predTest)
                    predTest = pd.DataFrame(predTest, columns=['predVal'])

                    survTrain = pd.concat([train_dataframes_small['survival'], val_dataframes['survival']], axis=0)
                    survTest = test_dataframes['survival']

                except:
                    predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                    predTest = pd.DataFrame({'predVal': np.full(survTest.shape[0], np.nan)})

                end_time = time.time()
                record_time = end_time - start_time
                print('Running Time: ' + str(record_time) + 'seconds')
                # fold_times.append(record_time)

                time_df = pd.DataFrame({
                    'dataset': [dataset],
                    'time_point': [current_time],
                    'fold': [fold],
                    'runtime_seconds': [record_time]
                })

                predTrain.index = survTrain.index
                predTest.index = survTest.index

                predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
                predTest = pd.concat([predTest, survTest[['time', 'status']]], axis=1)

                predTrain.to_csv(
                    resPath + "/SurvivalNet" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(
                        fold + 1) + '.csv', sep=',',
                    header=True)
                predTest.to_csv(resPath + "/SurvivalNet" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(
                    fold + 1) + '.csv', sep=',',
                                header=True)
                time_df.to_csv(
                    timerecPath + "/SurvivalNet" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold+1) + '.csv',
                    sep=',', header=True)

                del train_dataframes, test_dataframes, train_dataframes_small, val_dataframes, train_dict, val_dict, train_set, val_set, pretrain_set, test_array
                clear_memory()

        del common_dataList, survival_df, dataList
        clear_memory()
