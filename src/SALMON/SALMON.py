import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
# from torch.utils.checkpoint import checkpoint

import gc

class SALMON(nn.Module):
    def __init__(self, length_of_data, label_dim):
        super(SALMON, self).__init__()
        
        self.length_of_data = length_of_data
        hidden1 = 8
        hidden2 = 4
        
        hidden_cnv, hidden_clinical = length_of_data['cnv'], length_of_data['clinical']
        self.encoder1 = nn.Sequential(nn.Linear(length_of_data['mRNATPM'], hidden1),nn.Sigmoid())
        self.encoder2 = nn.Sequential(nn.Linear(length_of_data['miRNA'], hidden2),nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(hidden1 + hidden2 + \
                                            hidden_cnv + hidden_clinical, label_dim),nn.Sigmoid())

    def forward(self, x):
        x_d = None

        code1 = self.encoder1(x[:,0:self.length_of_data['mRNATPM']])
        code2 = self.encoder2(x[:,self.length_of_data['mRNATPM']: (self.length_of_data['mRNATPM'] + self.length_of_data['miRNA'])])
        lbl_pred = self.classifier(torch.cat((code1, code2, x[:, (self.length_of_data['mRNATPM'] + self.length_of_data['miRNA']):]), 1)) # predicted label
        code = torch.cat((code1, code2), 1)
            
        return x_d, code, lbl_pred

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.5)

def train(datasets, num_epochs, batch_size, learning_rate,
                        lambda_1, length_of_data, cuda, verbose):

    x = datasets['x']
    e = datasets['e']
    t = datasets['t']

    X = torch.FloatTensor(x)
    OS_event = torch.LongTensor(e)
    OS = torch.FloatTensor(t)

    dataloader = DataLoader(X, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False)
    lblloader = DataLoader(OS_event, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False)
    OSloader = DataLoader(OS, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False)

    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)

    model = SALMON(length_of_data=length_of_data, label_dim=1)

    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_nn_sum = 0
        iter = 0
        gc.collect()
        for data, lbl, survtime in zip(dataloader, lblloader, OSloader):
            optimizer.zero_grad() # zero the gradient buffer
            graph = data
            if cuda:
                model = model.cuda()
                graph = graph.cuda()
                lbl = lbl.cuda()
            # ===================forward=====================
            output, code, lbl_pred = model(graph)
            # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
            # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
            current_batch_len = len(survtime)
            R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
            for i in range(current_batch_len):
                for j in range(current_batch_len):
                    R_matrix_train[i,j] = survtime[j] >= survtime[i]

            train_R = torch.FloatTensor(R_matrix_train)
            if cuda:
                train_R = train_R.cuda()
            train_ystatus = lbl

            theta = lbl_pred.reshape(-1)
            exp_theta = torch.exp(theta)

            loss_nn = -torch.mean( (theta - torch.log(torch.sum( exp_theta*train_R ,dim=1))) * train_ystatus.float() )

            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

            loss = loss_nn + lambda_1 * l1_reg
            if verbose > 0:
                print("\nloss_nn: %.4f, L1: %.4f" % (loss_nn, lambda_1 * l1_reg))
            loss_nn_sum = loss_nn_sum + loss_nn.data.item()
            # ===================backward====================
            loss.backward()
            optimizer.step()

            iter += 1
            torch.cuda.empty_cache()
    return(model)

def predict(model, newdata, cuda):
    data = torch.FloatTensor(newdata.to_numpy())
    graph = Variable(data)
    if cuda:
        model = model.cuda()
        graph = graph.cuda()
        # lbl = lbl.cuda()
    output, code, lbl_pred = model(graph)
    pred = lbl_pred.data.cpu().numpy().reshape(-1)
    pred = pd.DataFrame(pred)
    return pred