import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/TF-LogHazardNet")

import os
from helper_process import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/TF-ProcessData'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/TF-LogHazardNet") == False:
#     os.makedirs(resPath + "/TF-LogHazardNet")

def run(datPath, resPath):
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/TF-LogHazardNet/" + fold) == False:
            os.makedirs(resPath + "/TF-LogHazardNet/" + fold)

        dataTypesUsed = ["omicsinter", "clinical", "survival"]

        for seed in range(1, 11):
            print ("****** seed " + str(seed) + " ******")

            train_dataframes = {}
            for dataType in dataTypesUsed:
                df = pd.read_csv(datPath + "/" + fold + "/" + dataType + "_train_" + str(seed) + ".csv", sep = ',', header=0, index_col=0)
                train_dataframes[dataType] = df

            test_dataframes = {}
            for dataType in dataTypesUsed:
                df = pd.read_csv(datPath + "/" + fold + "/" + dataType + "_val_" + str(seed) + ".csv", sep = ',', header=0, index_col=0)
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

            predTrain.index = surv_train.index
            predTest.index = surv_test.index

            predTrain = pd.concat([predTrain, surv_train], axis=1)
            predTest = pd.concat([predTest, surv_test], axis=1)

            predTrain.to_csv(resPath + "/TF-LogHazardNet" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predTest.to_csv(resPath + "/TF-LogHazardNet" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)

