import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/CSAE")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
from utils import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

if os.path.exists(resPath + "/CSAE") == False:
    os.makedirs(resPath + "/CSAE")

all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]

for fold in all_folds:
    print(fold)
    if os.path.exists(resPath + "/CSAE/" + fold) == False:
        os.makedirs(resPath + "/CSAE/" + fold)

    dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
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

        survTrain = train_dataframes['survival']
        survTest = test_dataframes['survival']

        del train_dataframes['survival']
        del test_dataframes['survival']

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

        predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
        predVal = pd.concat([predVal, survTest[['time', 'status']]], axis=1)

        predTrain.to_csv(resPath + "/CSAE" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                         header=True)
        predVal.to_csv(resPath + "/CSAE" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                       header=True)