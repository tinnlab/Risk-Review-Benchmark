import sys
sys.path.append("./")

import os
from functools import reduce
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from helper_process import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/MDNNMD") == False:
#     os.makedirs(resPath + "/MDNNMD")

def run(datPath, resPath, timerecPath):
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
               "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
               "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
               "TCGA-STAD", "TCGA-UCEC"]

    for dataset in alldatasets:
        print(dataset)
        if os.path.exists(resPath + "/MDNNMD/" + dataset) == False:
            os.makedirs(resPath + "/MDNNMD/" + dataset)
        if os.path.exists(timerecPath + "/MDNNMD/" + dataset) == False:
            os.makedirs(timerecPath + "/MDNNMD/" + dataset)

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
            print('Running Time: {}'.format(current_time))

            if os.path.exists(os.path.join(resPath, "MDNNMD", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(resPath, "MDNNMD", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join(timerecPath, "MDNNMD", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(timerecPath, "MDNNMD", dataset, 'Time' + str(current_time)))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)
            # fold_times = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
                print('Running Fold: {}'.format(fold))
                train_dataframes = {name: df.iloc[train_idx]
                                    for name, df in common_dataList.items()}

                test_dataframes = {name: df.iloc[val_idx]
                                for name, df in common_dataList.items()}

                surv_train = train_dataframes['survival']
                surv_train_class = surv_train.copy()
                surv_train_class.loc[:, "status"] = 1 - surv_train.loc[:,
                                                        "status"].to_numpy()  ## this is how MDNNMD define the classes
                surv_test = test_dataframes['survival']

                alive_train = surv_train[surv_train['status'] == 0].index.tolist()
                dead_train = surv_train[surv_train['status'] == 1].index.tolist()

                train_dataframes, stats, scalers = process_traindf(train_dataframes)
                test_dataframes = process_testdf(test_dataframes, stats, scalers)

                ## select features for each omics type
                train_dataframes, test_dataframes = featselect_df(train_dataframes, surv_train_class, test_dataframes)

                ## after data processing, further split the train data into train set and val set
                alive_train, alive_val = train_test_split(alive_train, test_size=0.2, random_state=1234)
                dead_train, dead_val = train_test_split(dead_train, test_size=0.2, random_state=1234)

                train_patients = alive_train + dead_train
                val_patients = alive_val + dead_val

                train_dataframes_small = {name: df.loc[train_patients]
                                        for name, df in train_dataframes.items()}
                val_dataframes = {name: df.loc[val_patients]
                                for name, df in train_dataframes.items()}

                surv_train_class_small = surv_train_class.loc[train_patients]
                surv_val = surv_train_class.loc[val_patients]

                ## begin training
                allres_fulltrain = {}
                allres_val = {}
                allres_test = {}

                ## configs
                allconfigs = {}
                allconfigs['hidden_units'] = {'mRNATPM': [1000, 500, 500, 100], 'cnv': [1000, 500, 500, 100],
                                            'clinical': [1000, 1000, 1000, 100]}
                allconfigs['learning_rate'] = {'mRNATPM': 0.00001, 'cnv': 0.00001, 'clinical': 0.001}
                allconfigs['epoch'] = {'mRNATPM': 40, 'cnv': 40, 'clinical': 60}
                allconfigs['drop'] = {'mRNATPM': 0.0, 'cnv': 0.0, 'clinical': 1.0}
                allconfigs['batch_size'] = {'mRNATPM': 32, 'cnv': 32, 'clinical': 32}

                start_time = time.time()

                for name, df in train_dataframes_small.items():
                    learning_rate = allconfigs['learning_rate'][name]
                    drop = allconfigs['drop'][name]
                    hidden_units = allconfigs['hidden_units'][name]
                    epoch = allconfigs['epoch'][name]
                    batch_size = allconfigs['batch_size'][name]

                    try:
                        tf.reset_default_graph()
                        mod = train(df, surv_train_class_small, hidden_units, learning_rate, epoch, drop, batch_size)

                        fulltraindat = train_dataframes[name]
                        valdat = val_dataframes[name]
                        testdat = test_dataframes[name]

                        predres_fulltrain = predict(mod, fulltraindat)
                        predres_val = predict(mod, valdat)
                        predres_test = predict(mod, testdat)

                        allres_fulltrain[name] = predres_fulltrain
                        allres_val[name] = predres_val
                        allres_test[name] = predres_test

                    except:
                        mod = "An exception occurred"
                        predfullTrain = pd.DataFrame({'predTrain': np.full(surv_train.shape[0], np.nan)})
                        predTest = pd.DataFrame({'predVal': np.full(surv_test.shape[0], np.nan)})
                        break

                if type(mod) != str:
                    opt_weights = find_optimal_weights(allres_val, surv_val['status'].to_numpy())
                    opt_weights = opt_weights['optimal_weights']
                    predfullTrain = np.full(surv_train.shape[0], 0)
                    predTest = np.full(surv_test.shape[0], 0)
                    for name in opt_weights.keys():
                        scorefullTrain = opt_weights[name] * allres_fulltrain[name]
                        predfullTrain = predfullTrain + scorefullTrain
                        scoreTest = opt_weights[name] * allres_test[name]
                        predTest = predTest + scoreTest
                    predfullTrain = pd.DataFrame({'predTrain': predfullTrain})
                    predTest = pd.DataFrame({'predVal': predTest})

                end_time = time.time()
                record_time = end_time - start_time
                # fold_times.append(record_time)
                print('Total Running Time: ' + str(record_time))

                time_df = pd.DataFrame({
                    'dataset': [dataset],
                    'time_point': [current_time],
                    'fold': [fold],
                    'runtime_seconds': [record_time]
                })


                predfullTrain.index = surv_train.index
                predTest.index = surv_test.index

                predfullTrain = pd.concat([predfullTrain, surv_train[['time', 'status']]], axis=1)
                predTest = pd.concat([predTest, surv_test[['time', 'status']]], axis=1)

                predfullTrain.to_csv(resPath + "/MDNNMD" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv', sep=',',
                                    header=True)
                predTest.to_csv(resPath + "/MDNNMD" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv', sep=',',
                                header=True)
                time_df.to_csv(
                    timerecPath + "/MDNNMD" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv',
                    sep=',', header=True)