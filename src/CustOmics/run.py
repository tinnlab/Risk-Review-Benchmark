import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/CustOmics")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np

from src.network.customics import CustOMICS
from helper_process import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

device = torch.device('cpu')

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/CustOmics") == False:
#     os.makedirs(resPath + "/CustOmics")

def run(datPath, resPath):
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    # all_folds = ["TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    #     "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
    #     "TCGA-STAD", "TCGA-UCEC"]

    # all_folds = ["TCGA-CESC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/CustOmics/" + fold) == False:
            os.makedirs(resPath + "/CustOmics/" + fold)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "cnv", "meth450", "clinical", "survival"]

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
        common_dataList = filter_df(common_dataList)

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
            survTrain = pd.concat([survTrain, train_dataframes['clinical'].loc[:, ['history_other_malignancy__no']]], axis=1)
            survVal = test_dataframes['survival']

            train_dataframes, scalers = process_traindf(train_dataframes)
            test_dataframes = process_testdf(test_dataframes, scalers)
            test_dataframes = {key: test_dataframes[key] for key in train_dataframes.keys() if key in test_dataframes}

            x_dim = [train_dataframes[omic_source].shape[1] for omic_source in train_dataframes.keys()]

            ## not all datasets incorporate the tumor type/ cancer stage information
            ## so I use history_other_malignancy__no for the classification module
            label = 'history_other_malignancy__no'    # labels for classification problems
            event = 'status'
            surv_time = 'time'
            task = 'survival'
            sources = list(train_dataframes)

            ## set up hyperparameters
            batch_size = 32
            n_epochs = 10

            hidden_dim = [512, 256]    # list of neurones for the hidden layers of the intermediate autoencoders
            central_dim = [512, 256]    # list of neurones for the hidden layers of the central autoencoder
            rep_dim = 128  #
            latent_dim = 128    # size of the latent vector
            num_classes = 2    # number of classes for classification problem
            dropout = 0.2    # dropout rate
            beta = 1   # value of the regulation coefficient for the beta-VAE
            lambda_classif = 5   # weight of the classification loss
            classifier_dim = [128, 64]   # list of neurones for the classifier hidden layers
            lambda_survival = 5   # weight of the survival loss
            survival_dim = [64, 32]   # list of neurones for the survival hidden layers
            train_params = {'switch': 5, 'lr': 1e-3}

            source_params = {}
            central_params = {'hidden_dim': central_dim, 'latent_dim': latent_dim, 'norm': True, 'dropout': dropout,
                            'beta': beta}
            classif_params = {'n_class': num_classes, 'lambda': lambda_classif, 'hidden_layers': classifier_dim,
                            'dropout': dropout}
            surv_params = {'lambda': lambda_survival, 'dims': survival_dim, 'activation': 'SELU', 'l2_reg': 1e-2,
                        'norm': True, 'dropout': dropout}
            for i, source in enumerate(sources):
                source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': rep_dim,
                                        'norm': True, 'dropout': 0.2}

            model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                            surv_params=surv_params, train_params=train_params, device=device).to(device)
            try:
                model.fit(omics_train=train_dataframes, clinical_df=survTrain, label=label, event=event, surv_time=surv_time,
                    batch_size=batch_size, n_epochs=n_epochs, verbose=True)
                predTrain = model.predict_risk(train_dataframes)
                predTrain = np.exp(predTrain)
                predTrain = pd.DataFrame(predTrain, columns=['predTrain'])
                predVal = model.predict_risk(test_dataframes)
                predVal = np.exp(predVal)
                predVal = pd.DataFrame(predVal, columns=['predVal'])
            except:
                predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                predVal = pd.DataFrame({'predVal': np.full(survVal.shape[0], np.nan)})

            predTrain.index = survTrain.index
            predVal.index = survVal.index

            predTrain = pd.concat([predTrain, survTrain], axis=1)
            predVal = pd.concat([predVal, survVal], axis=1)

            predTrain.to_csv(resPath + "/CustOmics" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predVal.to_csv(resPath + "/CustOmics" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)