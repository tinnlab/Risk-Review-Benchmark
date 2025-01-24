import sys
sys.path.append("./")

import gc
import time
import collections
from functools import reduce
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
from utils import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/GDP") == False:
#     os.makedirs(resPath + "/GDP")

def clear_memory():
    """Function to aggressively clear memory"""
    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(20)  # Give system time to clean up

def run(datPath, resPath, timerecPath):
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
                   "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
                   "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
                   "TCGA-STAD", "TCGA-UCEC"]

    for dataset in alldatasets:
        print(dataset)

        if not os.path.exists(f"{resPath}/GDP/{dataset}"):
            os.makedirs(f"{resPath}/GDP/{dataset}")
        if not os.path.exists(f"{timerecPath}/GDP/{dataset}"):
            os.makedirs(f"{timerecPath}/GDP/{dataset}")

        dataTypes = ["mRNATPM_map", "cnv_map", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM_map", "cnv_map", "clinical", "survival"]

        dataList = {}
        for dataType in dataTypes:
            df = pd.read_csv(f"{datPath}/{dataset}/{dataType}.csv", header=0, index_col=0)
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

        for current_time in range(1, 6):
            # if current_time != 1:
            #     continue
            print(f'Running Time: {current_time}')

            if not os.path.exists(os.path.join(resPath, "GDP", dataset, 'Time' + str(current_time))):
                os.makedirs(os.path.join(resPath, "GDP", dataset, 'Time' + str(current_time)))
            if not os.path.exists(os.path.join(timerecPath, "GDP", dataset, 'Time' + str(current_time))):
                os.makedirs(os.path.join(timerecPath, "GDP", dataset, 'Time' + str(current_time)))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)

            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
                print(f'Running Fold: {fold}')

                train_dataframes = {name: df.iloc[train_idx]
                                    for name, df in common_dataList.items()}

                test_dataframes = {name: df.iloc[val_idx]
                                for name, df in common_dataList.items()}

                train_dataframes, scalers = process_traindf(train_dataframes)
                test_dataframes = process_testdf(train_dataframes, test_dataframes, scalers)

                survTrain = train_dataframes['survival']
                survVal = test_dataframes['survival']

                group = assign_column_groups(train_dataframes['mRNATPM_map'], train_dataframes['cnv_map'],
                                            train_dataframes['clinical'])

                datasets = {"train": {}}
                datasets["train"]['X'] = pd.concat([train_dataframes['mRNATPM_map'], train_dataframes['cnv_map'],
                                                    train_dataframes['clinical']], axis=1).to_numpy()
                datasets["train"]['O'] = train_dataframes['survival'].loc[:, "status"].to_numpy()
                datasets["train"]['T'] = train_dataframes['survival'].loc[:, "time"].to_numpy()

                train = ld.DataSet(datasets["train"]['X'], datasets["train"]['T'], datasets["train"]['O'], group)
                Datasets = collections.namedtuple('Datasets', ['train'])
                TrainDataset = Datasets(train=train)

                batch_size = find_divisors(len(survTrain.index))
                batch_size = find_larger_numbers(batch_size, round(len(survTrain.index) / 10))
                batch_size = batch_size[0]

                GDPmod = 1
                start_time = time.time()

                try:
                    fit(TrainDataset, batch_size, dataset, current_time, fold)
                except:
                    GDPmod = "An exception occurred"
                    predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                    predVal = pd.DataFrame({'predVal': np.full(survVal.shape[0], np.nan)})

                if type(GDPmod) != str:
                    predTrain = predict(train_dataframes, group, dataset, current_time, fold)
                    predVal = predict(test_dataframes, group, dataset, current_time, fold)
                    predTrain.columns = ['predTrain']
                    predVal.columns = ['predVal']

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
                predVal.index = survVal.index

                predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
                predVal = pd.concat([predVal, survVal[['time', 'status']]], axis=1)

                predTrain.to_csv(
                    resPath + "/GDP" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv',
                    sep=',', header=True)
                predVal.to_csv(
                    resPath + "/GDP" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv',
                    sep=',', header=True)
                time_df.to_csv(
                    timerecPath + "/GDP" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv',
                    sep=',', header=True)

                gc.collect()
                clear_memory()




