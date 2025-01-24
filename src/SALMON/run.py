import sys
sys.path.append("./")

import os
import time
from functools import reduce
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
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

def run(datPath, resPath, timerecPath):

    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    for dataset in alldatasets:
        print(dataset)
        if os.path.exists(resPath + "/SALMON/" + dataset) == False:
            os.makedirs(resPath + "/SALMON/" + dataset)
        if os.path.exists(timerecPath + "/SALMON/" + dataset) == False:
            os.makedirs(timerecPath + "/SALMON/" + dataset)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "miRNA", "cnv", "clinical", "survival"]

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
        common_dataList = { name: df for name, df in common_dataList.items() if name in dataTypesUsed }
        common_dataList = filter_df(common_dataList)

        ### split data in to train and test set
        survival_df = common_dataList['survival']
        
        for current_time in range(1, 6):
        # if current_time != 4:
        #     continue
            print(f'Running Time: {current_time}')

            if not os.path.exists(os.path.join(resPath, "SALMON", dataset, 'Time' + str(current_time))):
                os.makedirs(os.path.join(resPath, "SALMON", dataset, 'Time' + str(current_time)))
            if not os.path.exists(os.path.join(timerecPath, "SALMON", dataset, 'Time' + str(current_time))):
                os.makedirs(os.path.join(timerecPath, "SALMON", dataset, 'Time' + str(current_time)))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)
            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):

                print(f'Running Fold: {fold}')

                train_dataframes = {name: df.iloc[train_idx] for name, df in common_dataList.items()}
                test_dataframes = {name: df.iloc[val_idx] for name, df in common_dataList.items()}

                survTrain = train_dataframes['survival']
                survTest = test_dataframes['survival']

                train_dataframes, test_dataframes = process_df(train_dataframes, test_dataframes)

                ### begin training:
                # some hyperparameter
                num_epochs = 100
                batch_size = 256
                learning_rate_range = 10 ** np.arange(-4, -1, 0.3)
                lr = learning_rate_range[0]
                cuda = False
                verbose = 0
                measure_while_training = True
                lambda_1 = 1e-5  # L1

                length_of_data = {}
                for name, df in train_dataframes.items():
                    length_of_data[name] = df.shape[1]

                length_of_data = {key: length_of_data[key] for key in
                                ['mRNATPM', 'miRNA', 'cnv', 'clinical']}  ## this order is important

                inputDataTrain = pd.concat(train_dataframes, keys=train_dataframes.keys(), axis=1)
                inputDataTest = pd.concat(test_dataframes, keys=test_dataframes.keys(), axis=1)

                alltrainData = {'x': inputDataTrain.to_numpy(), 'e': survTrain.loc[:, "status"].to_numpy(),
                                't': survTrain.loc[:, "time"].to_numpy()}

                start_time = time.time()

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

                end_time = time.time()
                record_time = end_time - start_time
                print(f'Running Time: {record_time:.2f} seconds')

                time_df = pd.DataFrame({
                    'dataset': [dataset],
                    'time_point': [current_time],
                    'fold': [fold],
                    'runtime_seconds': [record_time]
                })

                predTrain.index = survTrain.index
                predVal.index = survTest.index

                predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
                predVal = pd.concat([predVal, survTest[['time', 'status']]], axis=1)

                predTrain.to_csv(
                    resPath + "/SALMON" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv',
                    sep=',', header=True)
                predVal.to_csv(
                    resPath + "/SALMON" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv',
                    sep=',', header=True)
                time_df.to_csv(
                    timerecPath + "/SALMON" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv',
                    sep=',', header=True)










