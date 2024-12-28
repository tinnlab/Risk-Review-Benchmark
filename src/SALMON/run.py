import sys
sys.path.append("./")
sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/SALMON")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
from SALMON import *

from utils import *

from rpy2.robjects.packages import importr, data
utils = importr('utils')

utils.install_packages('lmQCM', repos="https://cloud.r-project.org")



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.cuda.empty_cache()

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/SALMON") == False:
#     os.makedirs(resPath + "/SALMON")

def run(datPath, resPath):

    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/SALMON/" + fold) == False:
            os.makedirs(resPath + "/SALMON/" + fold)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "miRNA", "cnv", "clinical", "survival"]

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
        common_dataList = filter_df(common_dataList)

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

            train_dataframes, test_dataframes = process_df(train_dataframes, test_dataframes)

            ### begin training:
            # some hyperparameter
            num_epochs = 100
            batch_size = 256
            learning_rate_range = 10 ** np.arange(-4, -1, 0.3)
            lr = learning_rate_range[0]
            cuda = True
            verbose = 0
            measure_while_training = True
            lambda_1 = 1e-5  # L1

            length_of_data = {}
            for name, df in train_dataframes.items():
                length_of_data[name] = df.shape[1]

            length_of_data = {key: length_of_data[key] for key in ['mRNATPM', 'miRNA', 'cnv', 'clinical']}  ## this order is important

            inputDataTrain = pd.concat(train_dataframes, keys=train_dataframes.keys(), axis=1)
            inputDataTest = pd.concat(test_dataframes, keys=test_dataframes.keys(), axis=1)

            alltrainData = {'x': inputDataTrain.to_numpy(), 'e': survTrain.loc[:, "status"].to_numpy(),
                            't': survTrain.loc[:, "time"].to_numpy()}
            try:
                model = train(alltrainData, num_epochs, batch_size, lr, lambda_1, length_of_data, cuda, verbose)
            except:
                model = "An exception occurred"
                predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                predVal = pd.DataFrame({'predVal': np.full(survTest.shape[0], np.nan)})

            if type(model) != str:
                predTrain = predict(model, inputDataTrain, cuda).to_numpy()
                predVal = predict(model, inputDataTest, cuda).to_numpy()

                predTrain = np.exp(predTrain)
                predTrain = pd.DataFrame(predTrain, columns=['predTrain'])
                predVal = np.exp(predVal)
                predVal = pd.DataFrame(predVal, columns=['predVal'])

            predTrain.index = survTrain.index
            predVal.index = survTest.index

            predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
            predVal = pd.concat([predVal, survTest[['time', 'status']]], axis=1)

            predTrain.to_csv(resPath + "/SALMON" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predVal.to_csv(resPath + "/SALMON" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)










