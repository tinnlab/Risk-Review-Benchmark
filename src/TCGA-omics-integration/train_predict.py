import torch.nn as nn
import math
from collections import OrderedDict
import torch
import gc
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import random

# torch.cuda.empty_cache()
# torch.cuda.memory.empty_cache()
# torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


## set some hyperparameter
meta_lr = 1e-4
inner_lr = 1e-4
output_dim = 1


## maml model
class MAMLModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, hidden_features)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_features, hidden_features // 4)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(hidden_features // 4, hidden_features // 8)),
            ('relu3', nn.ReLU()),
            ('l4', nn.Linear(hidden_features // 8, out_features)),
        ]))
        self.out_features = out_features

    def forward(self, x):
        return self.model(x)

    def parameterised(self, x, weights):
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[6], weights[7])
        return x


### impute data
def impute_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    for name, df in dataframes.items():
        if name == 'survival':
            continue

        if name == 'mRNATPM':
            if df.max().max(skipna=True) > 100:
                df = np.log2(df + 1)

        if df.isna().any().any():
            # KNN Imputation with k=1
            imputer = KNNImputer(n_neighbors=1)
            imputed_data = imputer.fit_transform(df)
            # Reconstruct DataFrame with original index and columns
            df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
        processed_dataframes[name] = df
    return processed_dataframes

### compute pearson correlation using the covariance method
def fast_correlation_matrix(df):
    data = df.to_numpy()
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.dot(data_centered.T, data_centered) / (len(data) - 1)
    std_dev = np.sqrt(np.diag(cov_matrix))
    std_dev[std_dev == 0] = 1
    correlation_matrix = cov_matrix / np.outer(std_dev, std_dev)
    correlation_matrix = np.abs(correlation_matrix)
    return correlation_matrix

# process train data
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    scalers = {}
    processed_train_dataframes['survival'] = train_dataframes['survival']

    for name, df in train_dataframes.items():
        if name == 'survival':
            continue

        # Z-score transformation for tables different from survival
        scaler = StandardScaler()
        z_scored_data = scaler.fit_transform(df)  ## standardScaler automatically converts 0 sd to 1 sd.
        df_new = pd.DataFrame(z_scored_data, columns=df.columns, index=df.index)
        scalers[name] = scaler

        if name == 'clinical':
            for col in df_new.columns:
                if df[col].nunique() <= 5:
                    df_new[col] = df[col]

        else:
            # Remove features with over 0.7 Pearson correlation
            correlation_matrix = fast_correlation_matrix(df_new)
            upper = np.triu(correlation_matrix, k=1)
            to_drop = [column for column in range(correlation_matrix.shape[1]) if np.any(np.abs(upper[:, column]) > 0.7)]
            df_new = df_new.drop(df_new.columns[to_drop], axis=1)
        processed_train_dataframes[name] = df_new
    return (processed_train_dataframes, scalers)

# Now process test dataframes
def process_testdf(processed_train_dataframes, test_dataframes, scalers):
    processed_test_dataframes = {}
    processed_test_dataframes['survival'] = test_dataframes['survival']

    for name, df in test_dataframes.items():
        if name == 'survival':
            continue

        # Z-score transformation for tables different from survival
        # Use the scaler from training data if available
        z_scored_data = scalers[name].transform(df)
        df_new = pd.DataFrame(z_scored_data, columns=df.columns, index=df.index)

        if name == 'clinical':
            for col in df_new.columns:
                if df[col].nunique() <= 5:
                    df_new[col] = df[col]

        keep_columns = list(processed_train_dataframes[name].columns)
        df_new = df_new[keep_columns]
        processed_test_dataframes[name] = df_new
    return processed_test_dataframes


# https://github.com/gevaertlab/metalearning_survival
def calculate_cox_loss(df_in, theta):
    """
    :param df_in: dataframe of uncensored patients on whom cox loss is calculated, needed for getting the duration.
    :param theta: meta-learning model output.
    :return: cox hazard loss.
    """
    observed = df_in["status"].values
    observed = torch.FloatTensor(observed).to(device)

    df_in = df_in.reset_index()
    exp_theta = torch.exp(theta)
    exp_theta = torch.reshape(exp_theta, [exp_theta.shape[0]])
    theta = torch.reshape(theta, [theta.shape[0]])
    R_matrix_batch = np.zeros([exp_theta.shape[0], exp_theta.shape[0]], dtype=int)

    for i, row1 in df_in.iterrows():
        for j, row2 in df_in.iterrows():
            time_1 = row1["time"]
            time_2 = row2["time"]
            R_matrix_batch[i, j] = time_2 >= time_1

    R_matrix_batch = torch.FloatTensor(R_matrix_batch).to(device)

    loss = -(torch.sum(torch.mul(torch.sum(theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))),
                                 observed)) / len(observed))

    return loss


## train function
def fit(mat, source_tasks):
    torch.cuda.empty_cache()
    gc.collect()
    df_cox = mat
    n = math.ceil(df_cox.shape[0] * 0.2)
    input_dim = df_cox.shape[1] - 2
    hidden_dim = input_dim // 5
    target0_df_query = df_cox.iloc[:n]
    target0_df = df_cox[n:]

    def get_data(task, query=False):
        """
        :param task: cancer type.
        :param query: True if desired data is query set; False if support.
        :return: data containing desired cancer type and support/query set.
        """
        if task == source_tasks[0]:
            if query:
                return target0_df_query
            return target0_df

    class MAML():
        def __init__(self,
                     input_dim,
                     model,
                     inner_lr,
                     meta_lr,
                     K=1,
                     inner_steps=1,
                     tasks_per_meta_batch=1):

            self.model = model
            self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
            self.meta_optimiser = torch.optim.SGD(self.weights, meta_lr)
            # hyperparameters
            self.inner_lr = inner_lr
            self.meta_lr = meta_lr
            self.inner_steps = inner_steps
            self.tasks_per_meta_batch = tasks_per_meta_batch

            # metrics
            self.print_every = 1
            # self.concordance_index = []

        def get_duration(self, indices):
            return df_cox["time"].iloc[indices].values

        def inner_loop(self, task, test_time=False):
            # print(task)
            # reset inner model to current maml weights
            temp_weights = [w.clone() for w in self.weights]

            # perform training on data sampled from source task support set
            df_cox_task = get_data(task, False)
            df_in = df_cox_task.drop(["time", "status"], axis=1)
            X = torch.FloatTensor(df_in.values).to(device)

            for step in range(self.inner_steps):
                theta = self.model.parameterised(X, temp_weights)
                cox_loss = calculate_cox_loss(df_in=df_cox_task,
                                              theta=theta)

                # compute grad and update inner loop weights
                if test_time == False:
                    grad = torch.autograd.grad(cox_loss, temp_weights)
                    temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

            # sample new data for meta-update and compute loss
            df_cox_task_query = get_data(task, True)

            duration = df_cox_task_query["time"].values

            df_in = df_cox_task_query.drop(["time", "status"], axis=1)

            X = torch.FloatTensor(df_in.values).to(device)

            theta = self.model.parameterised(X, temp_weights)
            cox_loss = calculate_cox_loss(df_in=df_cox_task_query,
                                          theta=theta)

            # c_index = CIndex(theta, duration, df_cox_task_query["censorship"].values)
            # if (type(c_index) == str):
            #     c_index = 0
            #
            # if test_time:
            #     self.concordance_index.append(c_index)

            return cox_loss

        ### Dao wrote a new function for prediction
        def predict(self, df_in):
            temp_weights = [w.clone() for w in self.weights]
            # df_in = df.drop(["duration", "censorship"], axis=1)
            X = torch.FloatTensor(df_in.values).to(device)
            theta = self.model.parameterised(X, temp_weights)

            return theta

        ###

        def main_loop_source(self, num_iterations):
            epoch_loss = 0

            for iteration in range(1, num_iterations + 1):
                # compute meta loss
                meta_loss = 0

                for i in range(self.tasks_per_meta_batch):
                    randnum = random.randint(0, len(source_tasks) - 1)
                    task = source_tasks[randnum]
                    loss = self.inner_loop(task)

                    meta_loss += loss
                # print(loss)

                # compute meta gradient of loss with respect to maml weights
                meta_grads = torch.autograd.grad(meta_loss, self.weights)

                self.model.zero_grad()
                # assign meta gradient to weights and take optimisation step
                for w, g in zip(self.weights, meta_grads):
                    w.grad = g

                self.meta_optimiser.step()

                # log metrics
                epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
                if iteration % self.print_every == 0:
                    # print("Avg C-index: ", round(np.mean(self.concordance_index, ), 2))
                    # print("!")
                    print('iteration: ' + str(iteration))
                    epoch_loss = 0

        def main_loop_target(self, num_iterations, target_task, test_time=False):
            self.concordance_index = []
            epoch_loss = 0

            for iteration in range(1, num_iterations + 1):
                meta_loss = 0

                for i in range(self.tasks_per_meta_batch):
                    loss = self.inner_loop(target_task, test_time=True)
                    meta_loss += loss

                # compute meta gradient of loss with respect to maml weights
                if test_time == False:
                    meta_grads = torch.autograd.grad(meta_loss, self.weights)
                    self.model.zero_grad()
                    # assign meta gradient to weights and take optimisation step
                    for w, g in zip(self.weights, meta_grads):
                        w.grad = g

                    self.meta_optimiser.step()

                # log metrics
                epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
                if iteration % self.print_every == 0:  # self.print_every
                    cindex = round(np.mean(self.concordance_index), 2)
                    # print("\nAvg C-index: ", cindex)
                    # print("@")
                    # self.concordance_index.append(cindex)

                    epoch_loss = 0

    model = MAMLModel(input_dim, hidden_dim, output_dim).to(device)

    # Train on source tasks
    print("Starting training.")
    maml = MAML(input_dim, model, inner_lr=inner_lr, meta_lr=meta_lr)
    maml.main_loop_source(num_iterations=100)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': maml.meta_optimiser.state_dict(),
    # }, filepath)
    return maml


## prediction function
def predict(fit, mat):
    pred = fit.predict(mat)
    pred = pred.cpu().detach().numpy()
    return pred
