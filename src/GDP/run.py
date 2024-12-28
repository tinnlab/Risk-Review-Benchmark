import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/GDP")

import os
import load_data as ld
from itertools import repeat
import collections
from sklearn.model_selection import train_test_split
from functools import reduce
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

def run(datPath, resPath):
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/GDP/" + fold) == False:
            os.makedirs(resPath + "/GDP/" + fold)

        dataTypes = ["mRNATPM_map", "cnv_map", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM_map", "cnv_map", "clinical", "survival"]

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
        all_genes, top_genes = gene_filter(common_dataList)
        common_dataList = impute_df(common_dataList, all_genes, top_genes)

        ### split data in to train and test set
        survival_df = common_dataList['survival']
        alive_patients = survival_df[survival_df['status'] == 0].index.tolist()
        dead_patients = survival_df[survival_df['status'] == 1].index.tolist()

        for seed in range(1, 11):
            print("****** seed " + str(seed) + " ******")
            tf.reset_default_graph()
            alive_train, alive_test = train_test_split(alive_patients, test_size=0.2, random_state=seed)
            dead_train, dead_test = train_test_split(dead_patients, test_size=0.2, random_state=seed)
            train_patients = alive_train + dead_train
            test_patients = alive_test + dead_test

            train_dataframes = {name: df.loc[train_patients]
                                for name, df in common_dataList.items()}

            test_dataframes = {name: df.loc[test_patients]
                            for name, df in common_dataList.items()}

            train_dataframes, scalers = process_traindf(train_dataframes)
            test_dataframes = process_testdf(train_dataframes, test_dataframes, scalers)

            survTrain = train_dataframes['survival']
            survVal = test_dataframes['survival']

            group = assign_column_groups(train_dataframes['mRNATPM_map'], train_dataframes['cnv_map'], train_dataframes['clinical'])

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
            batch_size = batch_size[0]  ### need to add codes to find the even batch size

            GDPmod = 1
            try:
                fit(TrainDataset, batch_size, fold, seed)
            except:
                GDPmod = "An exception occurred"
                predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                predVal = pd.DataFrame({'predVal': np.full(survVal.shape[0], np.nan)})

            if type(GDPmod) != str:
                predTrain = predict(train_dataframes, group, fold, seed)
                predVal = predict(test_dataframes, group, fold, seed)
                predTrain.columns = ['predTrain']
                predVal.columns = ['predVal']

            predTrain.index = survTrain.index
            predVal.index = survVal.index

            predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
            predVal = pd.concat([predVal, survVal[['time', 'status']]], axis=1)

            predTrain.to_csv(resPath + "/GDP" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predVal.to_csv(resPath + "/GDP" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)

        clear_folder('./' + save_folder)
