import sys
sys.path.append("./")

import os
import time
from functools import reduce
from sklearn.model_selection import StratifiedKFold
from utils import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]


for dataset in alldatasets:
    print(dataset)
    
    if os.path.exists(resPath + "/CSAE/" + dataset) == False:
        os.makedirs(resPath + "/CSAE/" + dataset)
    if os.path.exists(timerecPath + "/CSAE/" + dataset) == False:
        os.makedirs(timerecPath + "/CSAE/" + dataset)

    dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
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
    common_dataList = impute_df(common_dataList)

    survival_df = common_dataList['survival']

    # all_times = {}
    for current_time in range(1, 6):
        # if current_time != 3:
        #     continue
        print(f'Running Time: {current_time}')
    
        if os.path.exists(os.path.join(resPath, "CSAE", dataset, 'Time' + str(current_time))) == False:
            os.makedirs(os.path.join(resPath, "CSAE", dataset, 'Time' + str(current_time)))
        if os.path.exists(os.path.join(timerecPath, "CSAE", dataset, 'Time' + str(current_time))) == False:
            os.makedirs(os.path.join(timerecPath, "CSAE", dataset, 'Time' + str(current_time)))

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)
        # fold_times = []
        for fold, (train_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
            # if fold != 1:
            #     continue
            print(f'Running Fold: {fold}')
            train_dataframes = {name: df.iloc[train_idx]
                                for name, df in common_dataList.items()}

            test_dataframes = {name: df.iloc[val_idx]
                               for name, df in common_dataList.items()}

            survTrain = train_dataframes['survival']
            survTest = test_dataframes['survival']

            del train_dataframes['survival']
            del test_dataframes['survival']

            start_time = time.time()

            try:
                mod = train(train_dataframes, survTrain)
            except:
                mod = "An exception occurred"
                predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                predVal = pd.DataFrame({'predVal': np.full(survTest.shape[0], np.nan)})

            if type(mod) != str:
                predTrain = predict(mod, train_dataframes)
                predVal = predict(mod, test_dataframes)

                predTrain.index = survTrain.index
                predTrain.columns = ['predTrain']
                predVal.index = survTest.index
                predVal.columns = ['predVal']

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

            predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
            predVal = pd.concat([predVal, survTest[['time', 'status']]], axis=1)

            predTrain.to_csv(resPath + "/CSAE" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv', sep=',',
                             header=True)
            predVal.to_csv(resPath + "/CSAE" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv', sep=',',
                           header=True)
            time_df.to_csv(
                timerecPath + "/CSAE" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(
                    fold) + '.csv',
                sep=',', header=True)