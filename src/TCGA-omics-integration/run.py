import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/TCGA-omics-integration")

import os
import gc
import time
from functools import reduce
from sklearn.model_selection import StratifiedKFold
from train_predict import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/TCGA-omics-integration") == False:
#     os.makedirs(resPath + "/TCGA-omics-integration")

def run(datPath, resPath, timerecPath):
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]
    # all_folds = ["TCGA-CESC"]

    for dataset in alldatasets:
        print(dataset)

        if os.path.exists(resPath + "/TCGA-omics-integration/" + dataset) == False:
            os.makedirs(resPath + "/TCGA-omics-integration/" + dataset)
        if os.path.exists(timerecPath + "/TCGA-omics-integration/" + dataset) == False:
            os.makedirs(timerecPath + "/TCGA-omics-integration/" + dataset)


        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "clinical", "survival"]

        dataList = {}
        for dataType in dataTypes:
            df = pd.read_csv(datPath + "/" + dataset + "/" + dataType + ".csv", header=0, index_col=0)
            dataList[dataType] = df

        common_rows = reduce(
            lambda x, y: x.intersection(y),
            [df.index for df in dataList.values()]
        )

        common_dataList = {name: df.loc[common_rows] for name, df in dataList.items()}
        common_dataList = {name: df for name, df in common_dataList.items() if name in dataTypesUsed}
        common_dataList = impute_df(common_dataList)

        survival_df = common_dataList['survival']

        # all_times = {}
        for current_time in range(1, 6):
        
            print(f'Running Time: {current_time}')

            if os.path.exists(os.path.join(resPath, "TCGA-omics-integration", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(resPath, "TCGA-omics-integration", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join(timerecPath, "TCGA-omics-integration", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(timerecPath, "TCGA-omics-integration", dataset, 'Time' + str(current_time)))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)
            # fold_times = []
            for fold, (train_idx, val_idx) in enumerate(
                    skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):

                print(f'Running Fold: {fold}')
                train_dataframes = {name: df.iloc[train_idx]
                                    for name, df in common_dataList.items()}

                test_dataframes = {name: df.iloc[val_idx]
                                for name, df in common_dataList.items()}

                train_dataframes, scaler = process_traindf(train_dataframes)
                test_dataframes = process_testdf(train_dataframes, test_dataframes, scaler)

                mat_train = pd.concat([train_dataframes['mRNATPM'], train_dataframes['clinical']], axis=1)

                correlation_matrix = fast_correlation_matrix(mat_train)
                upper = np.triu(correlation_matrix, k=1)
                to_drop = [column for column in range(correlation_matrix.shape[1]) if
                        np.any(np.abs(upper[:, column]) > 0.7)]
                mat_train = mat_train.drop(mat_train.columns[to_drop], axis=1)
                mat_train = pd.concat([mat_train, train_dataframes['survival']], axis=1)
                del correlation_matrix
                gc.collect()

                mat_test = pd.concat([test_dataframes['mRNATPM'], test_dataframes['clinical'],
                                    test_dataframes['survival']], axis=1)
                mat_test = mat_test.drop(mat_test.columns[to_drop], axis=1)

                start_time = time.time()

                try:
                    mod = fit(mat_train, [fold])
                except:
                    mod = "An exception occurred"
                    predTrain = pd.DataFrame({'predTrain': np.full(mat_train.shape[0], np.nan)})
                    predVal = pd.DataFrame({'predVal': np.full(mat_test.shape[0], np.nan)})

                if type(mod) != str:
                    input_train = mat_train.drop(["time", "status"], axis=1)
                    input_val = mat_test.drop(["time", "status"], axis=1)
                    predTrain = predict(mod, input_train)
                    predTrain = np.exp(predTrain)
                    predTrain = pd.DataFrame(predTrain, columns=['predTrain'])
                    predVal = predict(mod, input_val)
                    predVal = np.exp(predVal)
                    predVal = pd.DataFrame(predVal, columns=['predVal'])

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

                predTrain.index = mat_train.index
                predVal.index = mat_test.index

                predTrain = pd.concat([predTrain, mat_train[['time', 'status']]], axis=1)
                predVal = pd.concat([predVal, mat_test[['time', 'status']]], axis=1)

                predTrain.to_csv(resPath + "/TCGA-omics-integration" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv',
                                header=True)
                predVal.to_csv(resPath + "/TCGA-omics-integration" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv',
                            header=True)
                time_df.to_csv(
                    timerecPath + "/TCGA-omics-integration" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv',
                    sep=',', header=True)




