import sys
sys.path.append("./")
# sys.path.append("/data/daotran/Cancer_RP/Benchmark/ReviewPaper_MethodRun/SurvivalNet")

import os
from functools import reduce
from helper_process import *
from survivalnet.optimization import SurvivalAnalysis
import numpy as np
import pandas as pd
import cPickle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# datPath = '/nfs/blanche/share/daotran/SurvivalPrediction/AllData/ReviewPaper_Data'
# resPath = '/data/dungp/projects_catalina/risk-review/benchmark/run-results'

# if os.path.exists(resPath + "/SurvivalNet") == False:
#     os.makedirs(resPath + "/SurvivalNet")

def run(datPath, resPath):
    if not os.path.exists('./data'):
        os.makedirs('./data')

    all_folds = ["TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA",
        "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
        "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC",
        "TCGA-STAD", "TCGA-UCEC"]

    for fold in all_folds:
        print(fold)
        if os.path.exists(resPath + "/SurvivalNet/" + fold) == False:
            os.makedirs(resPath + "/SurvivalNet/" + fold)

        dataTypes = ["mRNATPM", "cnv", "miRNA", "meth450", "clinical", "survival"]
        dataTypesUsed = ["mRNATPM", "cnv", "clinical", "survival"]

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
        common_dataList = impute_df(common_dataList)

        ### split data in to train and test set
        survival_df = common_dataList['survival']
        alive_patients = survival_df[survival_df['status'] == 0].index.tolist()
        dead_patients = survival_df[survival_df['status'] == 1].index.tolist()

        for seed in range(1, 11):
            print("****** seed " + str(seed) + " ******")
            alive_train, alive_val, alive_test, dead_train, dead_val, dead_test = train_val_test_split(alive_patients, dead_patients, seed)

            train_patients = alive_train + dead_train
            val_patients = alive_val + dead_val
            test_patients = alive_test + dead_test

            train_dataframes = {name: df.loc[train_patients]
                                for name, df in common_dataList.items()}
            val_dataframes = {name: df.loc[val_patients]
                            for name, df in common_dataList.items()}
            test_dataframes = {name: df.loc[test_patients]
                            for name, df in common_dataList.items()}

            train_dataframes, scalers, cate_cols = process_traindf(train_dataframes)
            val_dataframes = process_val_test_df(val_dataframes, scalers, cate_cols)
            test_dataframes = process_val_test_df(test_dataframes, scalers, cate_cols)

            train_dict = transform_df(train_dataframes)
            val_dict = transform_df(val_dataframes)

            # Uses the entire dataset for pretraining
            pretrain_set = np.vstack((train_dict['X'], val_dict['X']))

            train_set = {}
            val_set = {}

            sa = SurvivalAnalysis()
            train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(
                train_dict['X'],
                train_dict['T'],
                train_dict['O']);
            val_set['X'], val_set['T'], val_set['O'], val_set['A'] = sa.calc_at_risk(
                val_dict['X'],
                val_dict['T'],
                val_dict['O']);

            # Writes data sets for bayesopt cost function's use.
            with file('./data/train_set', 'wb') as f:
                cPickle.dump(train_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
            with file('./data/val_set', 'wb') as f:
                cPickle.dump(val_set, f, protocol=cPickle.HIGHEST_PROTOCOL)

            # pretrain_config = {'pt_lr':0.01, 'pt_epochs':1000,
            #  				     'pt_batchsize':None,'corruption_level':.3}
            model, n_hidden, n_layers = fit(pretrain_set, train_set, val_set)
                                            # pretrain_config=pretrain_config, do_bayes_opt=False)

            ## prediction
            predTrain = predict(model, pretrain_set, n_hidden, n_layers)
            predTrain = np.exp(predTrain)
            predTrain = pd.DataFrame(predTrain, columns=['predTrain'])

            test_array = transform_df_pred(test_dataframes)
            predTest = predict(model, test_array, n_hidden, n_layers)
            predTest = np.exp(predTest)
            predTest = pd.DataFrame(predTest, columns=['predVal'])

            survTrain = pd.concat([train_dataframes['survival'], val_dataframes['survival']], axis=0)
            survTest = test_dataframes['survival']

            predTrain.index = survTrain.index
            predTest.index = survTest.index

            predTrain = pd.concat([predTrain, survTrain[['time', 'status']]], axis=1)
            predTest = pd.concat([predTest, survTest[['time', 'status']]], axis=1)

            predTrain.to_csv(resPath + "/SurvivalNet" + '/' + fold + '/Train_Res_' + str(seed) + '.csv', sep=',',
                            header=True)
            predTest.to_csv(resPath + "/SurvivalNet" + '/' + fold + '/Val_Res_' + str(seed) + '.csv', sep=',',
                        header=True)





