import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import torch
from neural_network import Graph_Survival_Analysis


## function to filter data
def fillim_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']

    for name, df in dataframes.items():
        if name == 'survival':
            continue

        if df.isnull().sum().sum() > 0:
            # Remove columns with missing values over 10%
            df = df[df.columns[df.isnull().mean() <= 0.1]]

            ## impute using weighted nearest neighbor
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )

        if df.max().max() > 100:
            df = np.log2(df + 1)
        processed_dataframes[name] = df

    return processed_dataframes


## process train data
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    processed_train_dataframes['survival'] = train_dataframes['survival']

    for name, df in train_dataframes.items():
        if name == 'survival':
            continue

        if name == 'mRNATPM':
            df = df[df.var().sort_values(ascending=False).head(6000).index]

        if name == 'miRNA':
            df = df[df.var().sort_values(ascending=False).head(600).index]

        scaler = StandardScaler()
        z_scored_data = scaler.fit_transform(df)  ## standardScaler automatically converts 0 sd to 1 sd.
        df_new = pd.DataFrame(z_scored_data, columns=df.columns, index=df.index)
        scalers[name] = scaler
        processed_train_dataframes[name] = df_new
    return (processed_train_dataframes, scalers)


## Now process test dataframes
def process_testdf(processed_train_dataframes, test_dataframes, scalers):
    processed_test_dataframes = {}
    processed_test_dataframes['survival'] = test_dataframes['survival']

    for name, df in test_dataframes.items():
        if name == 'survival':
            continue

        df = df[processed_train_dataframes[name].columns]
        z_scored_data = scalers[name].transform(df)
        df_new = pd.DataFrame(z_scored_data, columns=df.columns, index=df.index)
        processed_test_dataframes[name] = df_new
    return processed_test_dataframes


## calculate the W matrix
def calw_omics(tensor_data, param, k=10):
    sq_tensor1 = torch.unsqueeze(tensor_data, 1)  # N*1*d
    sq_tensor2 = torch.unsqueeze(tensor_data, 0)  # 1*N*d
    W_omics = ((sq_tensor1 - sq_tensor2) ** 2).sum(2)  # N*N*d -> N*N
    W_omics_temp = W_omics.reshape(-1, 1)
    distance = torch.median(W_omics_temp, 0)
    for i in range(W_omics.shape[0]):
        W_omics[i, :] = W_omics[i, :] / (param * distance[0])
    W_omics = torch.exp(-W_omics)
    if k > 0:
        topk, indices = torch.topk(W_omics, k)
        mask = torch.zeros_like(W_omics)
        mask = mask.scatter(1, indices, 1)
        W_omics = W_omics * mask
    return W_omics


## train function
def train(miRNA_feature, RNASeq_feature, RNASeq_train_tensor, miRNA_train_tensor,
          S, ytime, ystatus, train_lr, iter_max):
    model = Graph_Survival_Analysis(miRNA_feature.shape[1], RNASeq_feature.shape[1])

    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=train_lr, weight_decay=5e-4)
    iter = 0

    while iter < iter_max:
        iter = iter + 1
        model.train()
        index = np.squeeze(np.arange(0, RNASeq_feature.shape[0]))
        fir_index = np.random.choice(index, size=RNASeq_feature.shape[0], replace=False)
        ystatus_batch_train = ystatus[fir_index,]
        ystatus_train_tensor = torch.tensor(ystatus_batch_train, dtype=torch.float)

        ytime_batch_train = ytime[fir_index,]
        ytime_train_tensor = torch.tensor(ytime_batch_train, dtype=torch.float)

        real_batch_size = ystatus_train_tensor.shape[0]
        R_matrix_batch_train = torch.tensor(np.zeros([real_batch_size, real_batch_size], dtype=int),
                                            dtype=torch.float)

        for i in range(real_batch_size):
            R_matrix_batch_train[i,] = torch.tensor(
                np.array(list(map(int, (ytime_train_tensor >= ytime_train_tensor[i])))))
        model.train()
        theta = model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor, S)[fir_index,]
        exp_theta = torch.reshape(torch.exp(theta), [real_batch_size])
        theta = torch.reshape(theta, [real_batch_size])
        fuse_loss = -torch.mean(
            torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch_train), dim=1))),
                      torch.reshape(ystatus_train_tensor, [real_batch_size])))

        loss = fuse_loss
        # Update meta-parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('training done!')
    return model

