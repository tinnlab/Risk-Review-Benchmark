import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/TCGA-omics-integration")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
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

def run(datPath, resPath):
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]
    # all_folds = ["TCGA-CESC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/TCGA-omics-integration/" + fold) == False:
            os.makedirs(resPath + "/TCGA-omics-integration/" + fold)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "clinical", "survival"]

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
        common_dataList = { name: df for name, df in common_dataList.items() if name in dataTypesUsed }
        common_dataList = impute_df(common_dataList)

        ### split data in to train and test set
        survival_df = common_dataList['survival']
        alive_patients = survival_df[survival_df['status'] == 0].index.tolist()
        dead_patients = survival_df[survival_df['status'] == 1].index.tolist()

        for seed in range(1, 11):
            print ("****** seed " + str(seed) + " ******")
            alive_train, alive_test = train_test_split(alive_patients, test_size=0.2, random_state=seed)
            dead_train, dead_test = train_test_split(dead_patients,test_size=0.2, random_state=seed)
            train_patients = alive_train + dead_train
            test_patients = alive_test + dead_test

            train_dataframes = {name: df.loc[train_patients]
                for name, df in common_dataList.items()}

            test_dataframes = {name: df.loc[test_patients]
                for name, df in common_dataList.items()}

            train_dataframes, scaler = process_traindf(train_dataframes)
            test_dataframes = process_testdf(train_dataframes, test_dataframes, scaler)

            mat_train = pd.concat([train_dataframes['mRNATPM'], train_dataframes['clinical'],
                                train_dataframes['survival']], axis=1)
            mat_test = pd.concat([test_dataframes['mRNATPM'], test_dataframes['clinical'],
                                test_dataframes['survival']], axis=1)

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

            predTrain.index = mat_train.index
            predVal.index = mat_test.index

            predTrain = pd.concat([predTrain, mat_train[['time', 'status']]], axis=1)
            predVal = pd.concat([predVal, mat_test[['time', 'status']]], axis=1)

            predTrain.to_csv(resPath + "/TCGA-omics-integration" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',', header=True)
            predVal.to_csv(resPath + "/TCGA-omics-integration" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',', header=True)




