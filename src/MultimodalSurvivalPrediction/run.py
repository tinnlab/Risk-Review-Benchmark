import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/MultimodalSurvivalPrediction")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
from data_loader import MyDataset
import utils
from model import Model
from helper_process import *

# device = utils.test_gpu()
# torch.cuda.set_device(0)
# device = torch.device('cuda:0')  # Explicitly specify GPU 0

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/MultimodalSurvivalPrediction") == False:
#     os.makedirs(resPath + "/MultimodalSurvivalPrediction")

def run(datPath, resPath):
    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    # fold = "TCGA-LAML"
    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/MultimodalSurvivalPrediction/" + fold) == False:
            os.makedirs(resPath + "/MultimodalSurvivalPrediction/" + fold)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "cnv", "miRNA", "clinical", "survival"]

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

            train_dataframes, scalers = process_traindf(train_dataframes)
            test_dataframes = process_testdf(train_dataframes, test_dataframes, scalers)

            survTrain = train_dataframes['survival']
            survVal = test_dataframes['survival']

            ## after data processing, further split the train data into train set and val set
            alive_train, alive_val = train_test_split(alive_train, test_size=0.2, random_state=seed)
            dead_train, dead_val = train_test_split(dead_train, test_size=0.2, random_state=seed)

            train_patients = alive_train + dead_train
            val_patients = alive_val + dead_val

            train_dataframes_small = {name: df.loc[train_patients]
                                    for name, df in train_dataframes.items()}
            val_dataframes = {name: df.loc[val_patients]
                            for name, df in train_dataframes.items()}

            modality_dim = mod_dim(train_dataframes_small)
            keepcols = train_dataframes_small['clinical'].columns
            cate_cols, cate_embedding, n_continuous = find_cate(common_dataList['clinical'], keepcols)
            # cate_embedding = gen_embed_cate(common_dataList['clinical'], cate_cols)

            ## hyperparameters
            m_length = 128
            BATCH_SIZE = 16
            EPOCH = 30
            lr = 0.0005

            modalities = ['clinical', 'miRNA', 'mRNA', 'CNV']
            mydataset_train = MyDataset(modalities, train_dataframes_small, cate_cols)
            mydataset_val = MyDataset(modalities, val_dataframes, cate_cols)
            dataloaders = utils.get_dataloaders(mydataset_train, mydataset_val, BATCH_SIZE)

            survmodel = Model(modalities=modalities, m_length=m_length, dataloaders=dataloaders, fusion_method='attention',
                            trade_off=0.3, mode='total', input_modality_dim=modality_dim,
                            n_continuous=n_continuous, embedding_size=cate_embedding)  # only_cox
            # device=device)
            # Generate run tag
            run_tag = utils.compose_run_tag(model=survmodel, lr=lr, dataloaders=dataloaders, log_dir='.training_logs/',
                                            suffix='')

            fit_args = {
                'num_epochs': EPOCH,
                'lr': lr,
                'info_freq': 2,
                'log_dir': os.path.join('.training_logs/', run_tag),
                'lr_factor': 0.5,
                'scheduler_patience': 7,
            }
            try:
                # model fitting
                survmodel.fit(**fit_args)
                # torch.save(survmodel, './saved_model/modelSaved.pt')
                print('Training is done!')

                ## do prediction
                survmodel.test()
                # mydataset_test = MyDataset_test(modalities, train_dataframes)
                # dataloaders_test = utils.get_dataloaders_test(mydataset_test, BATCH_SIZE)
                data_prediction = gen_preddat(train_dataframes, train_dataframes_small, cate_cols)
                predTrain = survmodel.predict(data_prediction)
                predTrain = predTrain[0]['hazard'].detach().numpy()
                predTrain = np.exp(predTrain)
                predTrain = pd.DataFrame(predTrain, columns=['predTrain'])

                # mydataset_test = MyDataset_test(modalities, test_dataframes)
                # dataloaders_test = utils.get_dataloaders_test(mydataset_test, BATCH_SIZE)
                data_prediction = gen_preddat(test_dataframes, train_dataframes_small, cate_cols)
                predVal = survmodel.predict(data_prediction)
                predVal = predVal[0]['hazard'].detach().numpy()
                predVal = np.exp(predVal)
                predVal = pd.DataFrame(predVal, columns=['predVal'])
            except:
                predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                predVal = pd.DataFrame({'predVal': np.full(survVal.shape[0], np.nan)})

            predTrain.index = survTrain.index
            predVal.index = survVal.index

            predTrain = pd.concat([predTrain, survTrain], axis=1)
            predVal = pd.concat([predVal, survVal], axis=1)

            predTrain.to_csv(resPath + "/MultimodalSurvivalPrediction" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                                header=True)
            predVal.to_csv(resPath + "/MultimodalSurvivalPrediction" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                            header=True)

            print('Prediction is done!')
