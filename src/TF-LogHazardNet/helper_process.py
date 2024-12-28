import numpy as np
import pandas as pd
import copy
import torch
import torchtuples as tt  # Some useful functions
from pycox.models import LogisticHazard
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv


## define hyperparameters
dropout = 0.3
batchsize = 32
nodenum = 16

avglist = []
resultdf = []
bsvec = np.zeros(100)
interval = 100
grid = 100
droprate = dropout
n_batch = batchsize
n_node = nodenum


## evaluation function
def eval_func(surv_test, duration_test, event_test):
    # surv_test is the survival probability of patients across time
    # duration_test and event_test are the true observing time and status of patients
    ev = EvalSurv(surv_test, duration_test, event_test, censor_surv='km')
    testci = ev.concordance_td('antolini')

    # time_grid = np.array([12, 24, 36, 48, 60]) # calculate ibs for 5 years
    # ibs = ev.integrated_brier_score(time_grid)

    return testci


## train function
def fit(list_train, surv_train):
    df_train = pd.concat(list_train, axis=1)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1234)
    surv_train, surv_val = train_test_split(surv_train, test_size=0.2, random_state=1234)

    # expDat = pyreadr.read_r(datPath + test_dataset + "/exp.rds")[None]
    # x_test = np.array(copy.copy(expDat).astype('float32'))

    x_train = np.array(copy.copy(df_train).astype('float32'))
    x_val = np.array(copy.copy(df_val).astype('float32'))

    Y_train = np.array(surv_train["time"])
    E_train = np.array(surv_train["status"])
    Y_val = np.array(surv_val["time"])
    E_val = np.array(surv_val["status"])

    df_train["time"] = Y_train.astype('float32')
    df_train["status"] = E_train.astype('float32')
    df_val["time"] = Y_val.astype('float32')
    df_val["status"] = E_val.astype('float32')

    num_durations = interval
    labtrans = LogisticHazard.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_surv_train = labtrans.fit_transform(*get_target(df_train))
    y_surv_val = labtrans.transform(*get_target(df_val))

    train = (x_train, y_surv_train)
    val = (x_val, y_surv_val)

    # durations_test, events_test = get_target(df_test)
    durations_train, events_train = get_target(df_train)

    in_features = x_train.shape[1]
    out_features = labtrans.out_features

    net2 = torch.nn.Sequential(
        torch.nn.Linear(in_features, n_node),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(n_node),
        torch.nn.Dropout(droprate),

        torch.nn.Linear(n_node, n_node),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(n_node),
        torch.nn.Dropout(droprate),

        torch.nn.Linear(n_node, out_features)
    )

    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    model = LogisticHazard(net2, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

    batch_size = n_batch
    epochs = 100
    callbacks = [tt.cb.EarlyStopping()]

    model.fit(x_train, y_surv_train, batch_size, epochs, callbacks, val_data=val)
    return model


## predict function
def predict(model, newdatalist):
    df = pd.concat(newdatalist, axis=1)
    data = np.array(copy.copy(df).astype('float32'))
    pred = model.predict_surv_df(data)
    return pred