import sys
sys.path.append("./")

import time
import os
import pandas as pd
from helper_process import *


# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/TF-ProcessData'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/TF-LogHazardNet") == False:
#     os.makedirs(resPath + "/TF-LogHazardNet")

def run(datPath, resPath, timerecPath):
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]

    for dataset in alldatasets:
        print(dataset)

        if os.path.exists(resPath + "/TF-LogHazardNet/" + dataset) == False:
            os.makedirs(resPath + "/TF-LogHazardNet/" + dataset)
        if os.path.exists(timerecPath + "/TF-LogHazardNet/" + dataset) == False:
            os.makedirs(timerecPath + "/TF-LogHazardNet/" + dataset)

        dataTypesUsed = ["omicsinter", "clinical", "survival"]

        for current_time in range(1, 6):
            # if current_time != 2:
            #     continue

            print(f'Running Time: {current_time}')

            if os.path.exists(os.path.join(resPath, "TF-LogHazardNet", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(resPath, "TF-LogHazardNet", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join(timerecPath, "TF-LogHazardNet", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(timerecPath, "TF-LogHazardNet", dataset, 'Time' + str(current_time)))

            # fold_times = []
            for fold in range(1, 11):
                # if fold != 7:
                #     continue
                
                print(f'Running Fold: {fold}')

                train_dataframes = {}
                for dataType in dataTypesUsed:
                    df = pd.read_csv(datPath + "/" + dataset + "/" + 'Time' + str(current_time) + "/" + dataType + "_train_" + str(fold) + ".csv", sep = ',', header=0, index_col=0)
                    train_dataframes[dataType] = df

                test_dataframes = {}
                for dataType in dataTypesUsed:
                    df = pd.read_csv(datPath + "/" + dataset + "/" + 'Time' + str(current_time) + "/" + dataType + "_val_" + str(fold) + ".csv", sep = ',', header=0, index_col=0)
                    test_dataframes[dataType] = df

                test_dataframes['omicsinter'].columns = train_dataframes['omicsinter'].columns

                list_train = []
                for dataType in ['omicsinter', 'clinical']:
                    list_train.append(train_dataframes[dataType])

                list_test = []
                for dataType in ['omicsinter', 'clinical']:
                    list_test.append(test_dataframes[dataType])

                surv_train = train_dataframes['survival']
                surv_test = test_dataframes['survival']

                start_time = time.time()

                try:
                    mod = fit(list_train, surv_train)

                    predTrain = predict(mod, list_train).T
                    predTest = predict(mod, list_test).T

                    unique_time = [round(float(col)) for col in predTrain.columns]
                    column_names = [f"predTrain_{time}" for time in unique_time]
                    predTrain.columns = column_names

                    unique_time = [round(float(col)) for col in predTest.columns]
                    column_names = [f"predVal_{time}" for time in unique_time]
                    predTest.columns = column_names
                except:
                    time_list = list(range(0, surv_train['time'].max() + 10, 100))
                    column_names = [f"predTrain_{time}" for time in time_list]
                    predTrain = pd.DataFrame(np.nan,
                                            index=range(len(surv_train)),
                                            columns=column_names)

                    time_list = list(range(0, surv_test['time'].max() + 10, 100))
                    column_names = [f"predVal_{time}" for time in time_list]
                    predTest = pd.DataFrame(np.nan,
                                            index=range(len(surv_test)),
                                            columns=column_names)

                end_time = time.time()
                record_time = end_time - start_time
                # fold_times.append(record_time)
                print(f'Running Time: {record_time:.2f} seconds')

                time_df = pd.DataFrame({
                    'dataset': [dataset],
                    'time_point': [current_time],
                    'fold': [fold],
                    'runtime_seconds': [record_time]
                })

                predTrain.index = surv_train.index
                predTest.index = surv_test.index

                predTrain = pd.concat([predTrain, surv_train], axis=1)
                predTest = pd.concat([predTest, surv_test], axis=1)

                predTrain.to_csv(resPath + "/TF-LogHazardNet" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv',
                                header=True)
                predTest.to_csv(resPath + "/TF-LogHazardNet" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv',
                            header=True)
                time_df.to_csv(
                    timerecPath + "/TF-LogHazardNet" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv',
                    sep=',', header=True)

