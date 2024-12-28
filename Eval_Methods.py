import os
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings('ignore')

ResPath = '/nfs/blanche/share/daotran/SurvivalPrediction/Results'
# ResPath = "/data/dungp/projects_catalina/risk-review/benchmark/run-results"
# all_methods = ['IPF-LASSO', 'I-Boost', 'Priority-Lasso']
all_methods = os.listdir(ResPath)
# all_methods.remove('OmiEmbed')

## methods output Hazard Ratio
HR_methods = ['IPF-LASSO', 'M2EFM', 'Multimodal_NSCLC', 'SALMON', 'GDP', 'MiNet', 'SurvivalNet', 'SAE', 'CSAE', 'CustOmics', 'MultimodalSurvivalPrediction', 'FGCNSurv', 'I-Boost', 'Priority-Lasso', 'TCGA-omics-integration']
## methods output Survival Probability
# SP_methods = ['TF-LogHazardNet', 'TF-ESN', 'OmiEmbed', 'blockForest']
SP_methods = ['TF-LogHazardNet', 'TF-ESN', 'blockForest']

## methods output Vital Status with Probability
VP_methods = ['MDNNMD']


#### DEFINE FUNCTIONS
## function to find indices for specific times
def find_idx(times, all_times):
    idx_vec = []
    for time in times:
        temp_times = np.abs(all_times - time)
        idx = np.where(temp_times == np.min(temp_times))[0]
        idx_vec.append(idx[0])

    return idx_vec


## calculate the survival probability using hazard ratio
def cal_survprob(pred_train, pred_val):
    pred_train['log_HR'] = pred_train['predTrain'].apply(np.log)

    cph = CoxPHFitter()
    cph.fit(pred_train.loc[:, ['log_HR', 'time', 'status']], duration_col='time', event_col='status')
    baseline_hazard = cph.baseline_hazard_

    all_times = baseline_hazard.index.values
    max_time = np.max(all_times)
    target_times = np.linspace(0, max_time, 20).tolist()
    baseline_haz_indices = find_idx(target_times, all_times)
    baseline_haz = baseline_hazard.iloc[baseline_haz_indices, 0].values

    HR_val = pred_val['predVal'].to_numpy()
    CHF_val = HR_val[:, np.newaxis] * baseline_haz
    survprob_val = np.exp(-CHF_val)
    survprob_val = survprob_val.T
    survprob_val = pd.DataFrame(survprob_val)
    survprob_val.index = target_times

    return survprob_val


## calculate time-dependent CIndex and IBS
def cal_tdci_ibs(surv_test, duration_test, event_test):
    ev = EvalSurv(surv_test, duration_test, event_test, censor_surv='km')
    testci = ev.concordance_td('antolini')

    max_time = np.max(duration_test)
    time_grid = np.linspace(0, max_time, 20)

    ibs = ev.integrated_brier_score(time_grid)
    return testci, ibs


## calculate risk score using survival probability
def cal_ci(predRes):
    risk_score = -np.log(predRes)
    risk_score = np.sum(risk_score, axis=1)
    risk_score = risk_score / np.max(risk_score)
    return risk_score


#### CALCULATE METRICS FOR ALL METHODS
allRes = {}
for method in all_methods:
    all_datasets = os.listdir(os.path.join(ResPath, method))
    all_datasets = [x for x in all_datasets if "TCGA-" in x]
    all_datasets = sorted(all_datasets)  # Returns new sorted list
    allCIndex = []
    alltdCIndex = []
    allIBS = []

    for dataset in all_datasets:
        NewPath = os.path.join(ResPath, method, dataset)
        CIndex = []
        tdCIndex = []
        IBS = []

        for seed in range(1,11):
            TrainRes = pd.read_csv(os.path.join(ResPath, method, dataset) + "/Train_Res_" + str(seed) + ".csv", header=0, index_col=0)
            ValRes = pd.read_csv(os.path.join(ResPath, method, dataset) + "/Val_Res_" + str(seed) + ".csv", header=0, index_col=0)
            trueDat = np.array(ValRes[['time', 'status']])

            if method in HR_methods:
                try:
                    valSurvProb = cal_survprob(TrainRes, ValRes)
                    tmp_tdCI, tmp_ibs = cal_tdci_ibs(valSurvProb, trueDat[:, 0], trueDat[:, 1])
                except:
                    tmp_tdCI = np.NaN
                    tmp_ibs = np.NaN

                try:
                    tmp_CI = concordance_index_censored(event_indicator=ValRes['status'].values.astype(bool),
                                                     event_time=ValRes['time'].values,
                                                     estimate=ValRes['predVal'])[0]
                except:
                    tmp_CI = np.NaN



            if method in SP_methods:
                predVal = ValRes.drop(['time', 'status'], axis=1)
                try:
                    riskScore = cal_ci(predVal.to_numpy())
                    tmp_CI = concordance_index_censored(event_indicator=ValRes['status'].values.astype(bool),
                                                    event_time=ValRes['time'].values,
                                                    estimate=riskScore)[0]
                except:
                    tmp_CI = np.NaN
                time_points = [int(col.split('_')[-1]) for col in predVal.columns if '_' in col]
                valSurvProb = predVal.T
                valSurvProb = valSurvProb.reset_index(drop=True)
                valSurvProb.index = time_points

                try:
                    tmp_tdCI, tmp_ibs = cal_tdci_ibs(valSurvProb, trueDat[:, 0], trueDat[:, 1])
                except:
                    tmp_tdCI = np.NaN
                    tmp_ibs = np.NaN

            if method in VP_methods:
                try:
                    tmp_CI = concordance_index_censored(event_indicator=ValRes['status'].values.astype(bool),
                                                    event_time=ValRes['time'].values,
                                                    estimate=1-ValRes['predVal'].to_numpy())[0]
                except:
                    tmp_CI = np.NaN

                tmp_tdCI = np.NaN
                tmp_ibs = np.NaN

            CIndex.append(tmp_CI)
            tdCIndex.append(tmp_tdCI)
            IBS.append(tmp_ibs)

        CIndex = np.nanmean(CIndex)
        tdCIndex = np.nanmean(tdCIndex)
        IBS = np.nanmean(IBS)

        allCIndex.append(CIndex)
        alltdCIndex.append(tdCIndex)
        allIBS.append(IBS)

    allCIndex = np.nanmean(allCIndex)
    alltdCIndex = np.nanmean(alltdCIndex)
    allIBS = np.nanmean(allIBS)
    allRes[method] = {'CIndex':allCIndex, 'tdCIndex':alltdCIndex, 'IBS':allIBS}

Res_Table = pd.DataFrame.from_dict(allRes, orient='index')




#                                 CIndex  tdCIndex       IBS
# MiNet                         0.512108  0.225325  0.391150
# blockForest                   0.680203  0.623922  0.188269
# CustOmics                     0.594827  0.269098  0.377564
# MultimodalSurvivalPrediction  0.555638  0.251097  0.388994
# MDNNMD                        0.499810       NaN       NaN
# SurvivalNet                   0.548589  0.240665  0.390285
# M2EFM                         0.617260  0.154859  0.457580
# TF-LogHazardNet               0.601408  0.604506  0.256046
# Multimodal_NSCLC              0.655201  0.275329  0.386546
# IPF-LASSO                     0.660715  0.291036  0.399900
# SALMON                        0.512158  0.230631  0.386975
# TF-ESN                        0.556646  0.546491  0.222996
# I-Boost                       0.686833  0.294329  0.403774
# CSAE                          0.624946  0.281127  0.385540
# SAE                           0.612972  0.197399  0.382819
# GDP                           0.497878  0.225330  0.393232
# FGCNSurv                      0.607802  0.268492  0.386485
# Priority-Lasso                0.686829  0.307249  0.397283
# OmiEmbed                      0.517445  0.519984  0.229232
# TCGA-omics-integration        0.497655  0.238054  0.393065