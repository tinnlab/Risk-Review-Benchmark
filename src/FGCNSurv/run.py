import sys
sys.path.append("./")

import os
import time
from sklearn.model_selection import StratifiedKFold
from functools import reduce
from sklearn.model_selection import StratifiedKFold

from helper_process import *
import pandas as pd
import numpy as np
import torch
import random

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/FGCNSurv") == False:
#     os.makedirs(resPath + "/FGCNSurv")

def run(datPath, resPath, timerecPath):
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    "TCGA-STAD", "TCGA-UCEC"]

    for dataset in alldatasets:
        print(dataset)
        
        if os.path.exists(resPath + "/FGCNSurv/" + dataset) == False:
            os.makedirs(resPath + "/FGCNSurv/" + dataset)
        if os.path.exists(timerecPath + "/FGCNSurv/" + dataset) == False:
            os.makedirs(timerecPath + "/FGCNSurv/" + dataset)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "miRNA", "survival"]

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
        common_dataList = {name: df for name, df in common_dataList.items() if name in dataTypesUsed}
        common_dataList = fillim_df(common_dataList)

        survival_df = common_dataList['survival']

        # all_times = {}
        for current_time in range(1, 6):
            print(f'Running Time: {current_time}')

            if os.path.exists(os.path.join(resPath, "FGCNSurv", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(resPath, "FGCNSurv", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join(timerecPath, "FGCNSurv", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(timerecPath, "FGCNSurv", dataset, 'Time' + str(current_time)))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)
            # fold_times = []
            for fold, (train_idx, val_idx) in enumerate(
                    skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
                print(f'Running Fold: {fold}')
                train_dataframes = {name: df.iloc[train_idx]
                                    for name, df in common_dataList.items()}

                test_dataframes = {name: df.iloc[val_idx]
                                for name, df in common_dataList.items()}

                survTrain = train_dataframes['survival']
                survVal = test_dataframes['survival']

                train_dataframes, scalers = process_traindf(train_dataframes)
                test_dataframes = process_testdf(train_dataframes, test_dataframes, scalers)

                RNASeq_feature = np.array(train_dataframes['mRNATPM'])
                miRNA_feature = np.array(train_dataframes['miRNA'])

                clinical_feature = np.array(train_dataframes['survival'])
                ytime = np.squeeze(clinical_feature[:, 0])
                ystatus = np.squeeze(clinical_feature[:, 1])

                RNASeq_train_tensor = torch.tensor(RNASeq_feature, dtype=torch.float)
                miRNA_train_tensor = torch.tensor(miRNA_feature, dtype=torch.float)

                start_time = time.time()

                W_Gene = calw_omics(RNASeq_train_tensor, 0.3)
                W_miRNA = calw_omics(miRNA_train_tensor, 0.25)

                eps = np.finfo(float).eps
                W = (W_Gene + W_miRNA) / 2 + torch.eye(W_Gene.shape[0])
                D = torch.sum(W, dim=1)
                D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
                S = D_sqrt_inv * W * D_sqrt_inv

                random.seed(1)
                torch.manual_seed(1)
                np.random.seed(1)
                train_lr = 2e-4

                try:
                    model = train(miRNA_feature, RNASeq_feature, RNASeq_train_tensor, miRNA_train_tensor,
                                S, ytime, ystatus, train_lr, 100)

                    # prediction using training data
                    predTrain = model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor, S)
                    predTrain = predTrain.detach().numpy()
                    predTrain = np.exp(predTrain)
                    predTrain = pd.DataFrame(predTrain, columns=['predTrain'])

                    # prediction using validation data
                    RNASeq_ft_val = np.array(test_dataframes['mRNATPM'])
                    miRNA_ft_val = np.array(test_dataframes['miRNA'])

                    RNASeq_val_tensor = torch.tensor(RNASeq_ft_val, dtype=torch.float)
                    miRNA_val_tensor = torch.tensor(miRNA_ft_val, dtype=torch.float)

                    RNASeq_tensor = torch.cat((RNASeq_train_tensor, RNASeq_val_tensor), axis=0)
                    miRNA_tensor = torch.cat((miRNA_train_tensor, miRNA_val_tensor), axis=0)

                    W_Gene_all = calw_omics(RNASeq_tensor, 0.3)
                    W_miRNA_all = calw_omics(miRNA_tensor, 0.25)

                    # recalculate S
                    eps = np.finfo(float).eps
                    W = (W_Gene_all + W_miRNA_all) / 2 + torch.eye(W_Gene_all.shape[0])
                    D = torch.sum(W, dim=1)
                    D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
                    S = D_sqrt_inv * W * D_sqrt_inv

                    row = RNASeq_train_tensor.shape[0]
                    predVal = model.get_survival_result(RNASeq_tensor, miRNA_tensor, S)[row:, :]
                    predVal = predVal.detach().numpy()
                    predVal = np.exp(predVal)
                    predVal = pd.DataFrame(predVal, columns=['predVal'])
                except:
                    predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                    predVal = pd.DataFrame({'predVal': np.full(survVal.shape[0], np.nan)})

                end_time = time.time()
                record_time = end_time - start_time
                print(f'Running Time: {record_time:.2f} seconds')

                time_df = pd.DataFrame({
                    'dataset': [dataset],
                    'time_point': [current_time],
                    'fold': [fold],
                    'runtime_seconds': [record_time]
                })

                # fold_times.append(record_time)

                predTrain.index = survTrain.index
                predVal.index = survVal.index

                predTrain = pd.concat([predTrain, survTrain], axis=1)
                predVal = pd.concat([predVal, survVal], axis=1)

                predTrain.to_csv(resPath + "/FGCNSurv" + '/' + dataset + '/Time' + str(current_time) + '/Train_Res_' + str(fold) + '.csv', sep=',',
                                header=True)
                predVal.to_csv(resPath + "/FGCNSurv" + '/' + dataset + '/Time' + str(current_time) + '/Val_Res_' + str(fold) + '.csv', sep=',',
                            header=True)
                time_df.to_csv(
                    timerecPath + "/FGCNSurv" + '/' + dataset + '/Time' + str(current_time) + '/TimeRec_' + str(fold) + '.csv',
                    sep=',', header=True)

