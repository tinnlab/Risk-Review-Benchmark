import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/MiNet")

import os
from functools import reduce
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import random
from utils import *
from Train import train_omics_net, train_best_hyper, predict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# If using gpu:
# GPUID = random.choice([0, 1, 2, 3])
# torch.cuda.set_device(GPUID)

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'
# GenePwTable = pd.read_csv("/nfs/blanche/share/daotran/SurvivalPrediction/AllData/GenePwTable.csv", header=0, sep=",")

# if os.path.exists(resPath + "/MiNet") == False:
#     os.makedirs(resPath + "/MiNet")

def run(datPath, resPath, GenePwTable):

    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
                "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
                "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
                "TCGA-STAD", "TCGA-UCEC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/MiNet/" + fold) == False:
            os.makedirs(resPath + "/MiNet/" + fold)

        dataTypes = ["mRNATPM_map", "cnv_map", "miRNA", "meth450_map", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM_map", "meth450_map", "cnv_map", "clinical", "survival"]

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
        genes_overlapped = intersect_columns(common_dataList['mRNATPM_map'], common_dataList['cnv_map'], common_dataList['meth450_map'], GenePwTable)
        common_dataList = impute_df(common_dataList, genes_overlapped)

        ## incase some genes are removed from specific datasets, intersect the genes again
        df1 = common_dataList['mRNATPM_map']
        df2 = common_dataList['cnv_map']
        df3 = common_dataList['meth450_map']
        genes_overlapped = list(set(df1.columns) & set(df2.columns) & set(df3.columns))
        common_dataList['mRNATPM_map'] = df1[genes_overlapped]
        common_dataList['cnv_map'] = df2[genes_overlapped]
        common_dataList['meth450_map'] = df3[genes_overlapped]

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
            test_dataframes = process_testdf(test_dataframes, scalers)

            ## further split the train data into train set and val set
            alive_train, alive_val = train_test_split(alive_train, test_size=0.2, random_state=seed)
            dead_train, dead_val = train_test_split(dead_train, test_size=0.2, random_state=seed)

            train_patients = alive_train + dead_train
            val_patients = alive_val + dead_val

            train_dataframes_small = {name: df.loc[train_patients]
                                    for name, df in train_dataframes.items()}
            val_dataframes = {name: df.loc[val_patients]
                            for name, df in train_dataframes.items()}

            survTrain_small = train_dataframes_small['survival']
            clinTrain_small = train_dataframes_small['clinical']
            survVal = val_dataframes['survival']
            clinVal = val_dataframes['clinical']

            omicsTrain_small = gene_omics_train(train_dataframes_small)
            omicsVal = gene_omics_train(val_dataframes)

            # start = datetime.datetime.now()
            x_train, ytime_train, yevent_train, age_train = load_data(omicsTrain_small, clinTrain_small, survTrain_small)
            # end = datetime.datetime.now()
            # time_rec = end - start
            # time_rec = time_rec.seconds

            x_valid, ytime_valid, yevent_valid, age_valid = load_data(omicsVal, clinVal, survVal)

            ## generate data for prediction
            x_fulltrain, age_fulltrain, survTrain = pred_data_gene(train_dataframes)
            x_test, age_test, survTest = pred_data_gene(test_dataframes)

            genes_indices, pathway_indices = generate_id_array(omicsTrain_small, GenePwTable)
            pathway_indices = pathway_indices[:, np.argsort(pathway_indices[0])]

            ## begin training
            # hyperparameters
            In_Nodes = omicsTrain_small.shape[1]  ### number of omics
            Gene_Nodes = genes_indices[0, :].max() + 1 ### number of genes
            Pathway_Nodes = pathway_indices[0, :].max() + 1  ### number of pathways
            Hidden_Nodes = [22, 5]  ### number of hidden nodes
            # Hidden_Nodes = [64, 16]  ### number of hidden nodes

            ##### Initials
            max_epochs = 10000
            Drop_Rate = [0.7, 0.5]  ### dropout rates

            L2_Lambda = [0.01, 0.02, 0.04, 0.08, 0.10, 0.12]
            Initial_Learning_Rate = [1e-2, 5e-3, 1e-3]
            opt_cidx = 0.0

            for lr in Initial_Learning_Rate:
                for l2 in L2_Lambda:
                    # print("L2: ", l2, "LR: ", lr)
                    c_index_tr, c_index_va, epoch_return = train_omics_net(x_train, age_train, ytime_train, yevent_train, \
                                                                        x_valid, age_valid, ytime_valid, yevent_valid, \
                                                                        genes_indices, pathway_indices, \
                                                                        In_Nodes, Gene_Nodes, Pathway_Nodes,
                                                                        Hidden_Nodes, \
                                                                        lr, l2, max_epochs, Drop_Rate, step=100,
                                                                        tolerance=0.02, \
                                                                        sparse_coding=True)

                    # print("temporary results (cindex_Train, cindex_val, epoch): ", c_index_tr, c_index_va, epoch_return)
                    if (c_index_tr.item() > c_index_va.item()) and (c_index_va.item() > opt_cidx):
                        opt_l2 = l2
                        opt_lr = lr
                        opt_cidx_tr = c_index_tr
                        opt_cidx = c_index_va
                        opt_epoch = epoch_return

            try:
                model = train_best_hyper(x_train, age_train, ytime_train, yevent_train, \
                                    genes_indices, pathway_indices, \
                                    In_Nodes, Gene_Nodes, Pathway_Nodes,
                                    Hidden_Nodes, \
                                    opt_lr, opt_l2, opt_epoch, Drop_Rate, \
                                    sparse_coding=True)
            except:
                model = "An exception occurred"
                predTrain = pd.DataFrame({'predTrain': np.full(survTrain.shape[0], np.nan)})
                predTest = pd.DataFrame({'predVal': np.full(survTest.shape[0], np.nan)})

            if type(model) != str:
                predTrain = predict(model, x_fulltrain, age_fulltrain, genes_indices, pathway_indices, Drop_Rate)
                predTest = predict(model, x_test, age_test, genes_indices, pathway_indices, Drop_Rate)
                predTrain = pd.DataFrame({'predTrain': predTrain})
                predTest = pd.DataFrame({'predVal': predTest})

            predTrain.index = survTrain.index
            predTest.index = survTest.index

            predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
            predTest = pd.concat([predTest, survTest[['time', 'status']]], axis=1)

            predTrain.to_csv(resPath + "/MiNet" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predTest.to_csv(resPath + "/MiNet" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)













