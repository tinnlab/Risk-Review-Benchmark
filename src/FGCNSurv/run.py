import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/FGCNSurv")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
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

def run(datPath, resPath):
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/FGCNSurv/" + fold) == False:
            os.makedirs(resPath + "/FGCNSurv/" + fold)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "miRNA", "survival"]

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
        common_dataList = {name: df for name, df in common_dataList.items() if name in dataTypesUsed}
        common_dataList = fillim_df(common_dataList)

        ### split data in to train and test set
        survival_df = common_dataList['survival']
        alive_patients = survival_df[survival_df['status'] == 0].index.tolist()
        dead_patients = survival_df[survival_df['status'] == 1].index.tolist()

        for seed in range(1, 11):
            print("****** seed " + str(seed) + " ******")
            alive_train, alive_test = train_test_split(alive_patients, test_size=0.2, random_state=seed)
            dead_train, dead_test = train_test_split(dead_patients, test_size=0.2, random_state=seed)
            train_patients = alive_train + dead_train
            test_patients = alive_test + dead_test

            train_dataframes = {name: df.loc[train_patients]
                                for name, df in common_dataList.items()}

            test_dataframes = {name: df.loc[test_patients]
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

            predTrain.index = survTrain.index
            predVal.index = survVal.index

            predTrain = pd.concat([predTrain, survTrain], axis=1)
            predVal = pd.concat([predVal, survVal], axis=1)

            predTrain.to_csv(resPath + "/FGCNSurv" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predVal.to_csv(resPath + "/FGCNSurv" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)

