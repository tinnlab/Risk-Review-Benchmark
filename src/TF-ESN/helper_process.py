import numpy as np
import pandas as pd
import copy
import torch
from torch import nn
import torchtuples as tt  # Some useful functions
from pycox.models import LogisticHazard
from pycox.models.loss import NLLLogistiHazardLoss
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv


## define hyperparameters
lossweight = 0.5
encodenum = 16
encodesize = 16
netsize = 128
batch = 32
interval = 100

lw = lossweight
en = encodenum
ns = netsize
es = encodesize
ba = batch


class NetAESurv(nn.Module):
    def __init__(self, in_features, encoded_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, es * 3), nn.ReLU(),
            nn.Linear(es * 3, es * 2), nn.ReLU(),
            nn.Linear(es * 2, es * 2), nn.ReLU(),
            nn.Linear(es * 2, es * 2), nn.ReLU(),

            nn.Linear(es * 2, encoded_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_features, es * 2), nn.ReLU(),
            nn.Linear(es * 2, es * 2), nn.ReLU(),
            nn.Linear(es * 2, es * 2), nn.ReLU(),
            nn.Linear(es * 2, es * 3), nn.ReLU(),
            nn.Linear(es * 3, in_features),
        )
        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, ns * 3), nn.ReLU(),
            nn.Linear(ns * 3, ns * 3), nn.ReLU(),
            nn.Linear(ns * 3, ns * 2), nn.ReLU(),
            nn.Linear(ns * 2, ns * 2), nn.ReLU(),
            nn.Linear(ns * 2, ns), nn.ReLU(),
            nn.Linear(ns, out_features),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        phi = self.surv_net(encoded)
        return phi, decoded

    def predict(self, input):
        # Will be used by model.predict later.
        # As this only has the survival output,
        # we don't have to change LogisticHazard.
        encoded = self.encoder(input)
        return self.surv_net(encoded)


class LossAELogHaz(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss()  # NLLPMFLoss() #
        self.loss_ae = nn.MSELoss()

    def forward(self, phi, decoded, target_loghaz, target_ae):
        idx_durations, events = target_loghaz
        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_ae = self.loss_ae(decoded, target_ae)
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae


def eval_func(surv_test, duration_test, event_test):
    # surv_test is the survival probability of patients across time
    # duration_test and event_test are the true observing time and status of patients
    ev = EvalSurv(surv_test, duration_test, event_test, censor_surv='km')
    testci = ev.concordance_td('antolini')

    # time_grid = np.array([12, 24, 36, 48, 60]) # calculate ibs for 5 years
    # ibs = ev.integrated_brier_score(time_grid)
    return testci


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

    train = tt.tuplefy(x_train, (y_surv_train, x_train))
    # train = (x_train, y_surv)
    val = tt.tuplefy(x_val, (y_surv_val, x_val))

    # durations_test, events_test = get_target(df_test)
    durations_train, events_train = get_target(df_train)

    in_features = x_train.shape[1]
    encoded_features = en
    out_features = labtrans.out_features
    net = NetAESurv(in_features, encoded_features, out_features)

    loss = LossAELogHaz(lw)

    np.random.seed(1234)
    _ = torch.manual_seed(1234)
    model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts, loss=loss)

    metrics = dict(
        loss_surv=LossAELogHaz(1),
        loss_ae=LossAELogHaz(0)
    )
    callbacks = [tt.cb.EarlyStopping()]

    batch_size = ba
    epochs = 100

    model.fit(*train, batch_size, epochs, callbacks, val_data=val, metrics=metrics)

    return model


def predict(model, newdatalist):
    df = pd.concat(newdatalist, axis=1)
    data = np.array(copy.copy(df).astype('float32'))
    pred = model.predict_surv_df(data)
    return pred