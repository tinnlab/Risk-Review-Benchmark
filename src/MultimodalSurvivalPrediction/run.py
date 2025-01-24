import sys
sys.path.append("./")

import os
import gc
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import time
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

def clear_memory():
    """Function to aggressively clear memory"""
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(10)

def run(datPath, resPath, timerecPath):
    alldatasets = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
                    "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
                    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
                    "TCGA-STAD", "TCGA-UCEC"]
    
    device = torch.device('cuda:0')  # Explicitly specify GPU 0

    # fold = "TCGA-LAML"
    for dataset in alldatasets:
        print(dataset)
        if os.path.exists(resPath + "/MultimodalSurvivalPrediction/" + dataset) == False:
            os.makedirs(resPath + "/MultimodalSurvivalPrediction/" + dataset)
        if os.path.exists(timerecPath + "/MultimodalSurvivalPrediction/" + dataset) == False:
            os.makedirs(timerecPath + "/MultimodalSurvivalPrediction/" + dataset)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "cnv", "miRNA", "clinical", "survival"]

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
        common_dataList = filter_df(common_dataList)

        survival_df = common_dataList['survival']

        # all_times = {}
        for current_time in range(1, 6):
            print(f'Running Time: {current_time}')

            if os.path.exists(
                    os.path.join(resPath, "MultimodalSurvivalPrediction", dataset, 'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(resPath, "MultimodalSurvivalPrediction", dataset, 'Time' + str(current_time)))
            if os.path.exists(os.path.join(timerecPath, "MultimodalSurvivalPrediction", dataset,
                                        'Time' + str(current_time))) == False:
                os.makedirs(os.path.join(timerecPath, "MultimodalSurvivalPrediction", dataset, 'Time' + str(current_time)))

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=current_time)
            # fold_times = []
            for fold, (train_idx, val_idx) in enumerate(
                    skf.split(np.zeros(len(survival_df['status'])), survival_df['status']), 1):
                # if fold != 1:
                #     continue
                print(f'Running Fold: {fold}')

                # GPUID = random.choice(range(num_gpus))
                # torch.cuda.set_device(0)

                train_dataframes = {name: df.iloc[train_idx]
                                    for name, df in common_dataList.items()}

                test_dataframes = {name: df.iloc[val_idx]
                                for name, df in common_dataList.items()}

                train_dataframes, scalers = process_traindf(train_dataframes)
                test_dataframes = process_testdf(train_dataframes, test_dataframes, scalers)

                survTrain = train_dataframes['survival']
                survVal = test_dataframes['survival']

                alive_patients = survTrain[survTrain['status'] == 0].index.tolist()
                dead_patients = survTrain[survTrain['status'] == 1].index.tolist()

                ## after data processing, further split the train data into train set and val set
                alive_train, alive_val = train_test_split(alive_patients, test_size=0.2, random_state=1234)
                dead_train, dead_val = train_test_split(dead_patients, test_size=0.2, random_state=1234)

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

                survmodel = Model(modalities=modalities, m_length=m_length, dataloaders=dataloaders,
                                fusion_method='attention',
                                trade_off=0.3, mode='total', input_modality_dim=modality_dim,
                                n_continuous=n_continuous, embedding_size=cate_embedding, device=device)  # only_cox
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

                start_time = time.time()

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
                    predTrain = predTrain[0]['hazard'].detach().cpu().numpy()
                    predTrain = np.exp(predTrain)
                    predTrain = pd.DataFrame(predTrain, columns=['predTrain'])

                    # mydataset_test = MyDataset_test(modalities, test_dataframes)
                    # dataloaders_test = utils.get_dataloaders_test(mydataset_test, BATCH_SIZE)
                    data_prediction = gen_preddat(test_dataframes, train_dataframes_small, cate_cols)
                    predVal = survmodel.predict(data_prediction)
                    predVal = predVal[0]['hazard'].detach().cpu().numpy()
                    predVal = np.exp(predVal)
                    predVal = pd.DataFrame(predVal, columns=['predVal'])
                except:
                    predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                    predVal = pd.DataFrame({'predVal': np.full(survVal.shape[0], np.nan)})

                end_time = time.time()
                record_time = end_time - start_time
                # fold_times.append(record_time)
                print(f'Running Time: {record_time:.2f} seconds')

                time_df = pd.DataFrame({
                    'dataset': [dataset],
                    'time_point': [current_time],
                    'fold': [fold],
                    'runtime_seconds': [record_time]
                })

                predTrain.index = survTrain.index
                predVal.index = survVal.index

                predTrain = pd.concat([predTrain, survTrain], axis=1)
                predVal = pd.concat([predVal, survVal], axis=1)

                predTrain.to_csv(
                    resPath + "/MultimodalSurvivalPrediction" + '/' + dataset + '/Time' + str(
                        current_time) + '/Train_Res_' + str(fold) + '.csv',
                    sep=',',
                    header=True)
                predVal.to_csv(resPath + "/MultimodalSurvivalPrediction" + '/' + dataset + '/Time' + str(
                    current_time) + '/Val_Res_' + str(fold) + '.csv',
                            sep=',',
                            header=True)
                time_df.to_csv(
                    timerecPath + "/MultimodalSurvivalPrediction" + '/' + dataset + '/Time' + str(
                        current_time) + '/TimeRec_' + str(
                        fold) + '.csv',
                    sep=',', header=True)

                print('Prediction is done!')

                clear_memory()
