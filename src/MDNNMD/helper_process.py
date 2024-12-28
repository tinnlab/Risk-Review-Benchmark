from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import tensorflow as tf
import random, math
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def knn_impute(X, k=15):
    X = np.array(X, dtype=float)
    X_imputed = X.copy()
    na_columns = np.where(np.any(np.isnan(X), axis=0))[0]
    for col in na_columns:
        missing_idx = np.isnan(X[:, col])
        non_missing_idx = ~missing_idx

        if not np.any(missing_idx):
            continue
        if not np.any(non_missing_idx):
            raise ValueError("Cannot impute column with all missing values")

        for idx in np.where(missing_idx)[0]:
            row = X[idx]
            valid_features1 = ~np.isnan(row)
            valid_rows = np.where(non_missing_idx)[0]
            valid_features2 = ~np.any(np.isnan(X[valid_rows]), axis=0)
            valid_features = valid_features1 & valid_features2

            distances = cdist(row[np.newaxis, valid_features], X[valid_rows][:, valid_features])[0]
            k_nearest = valid_rows[np.argsort(distances)[:k]]
            X_imputed[idx, col] = np.mean(X[k_nearest, col])

    return X_imputed


## impute function
def impute_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        # if name == "clinical":
        #     filtered_columns = ["age_at_initial_pathologic_diagnosis", "history_other_malignancy__no"]
        #     df = df.loc[:, filtered_columns]

        else:
            if df.max().max(skipna=True) > 100:
                df = np.log2(df + 1)
            if df.isna().any().any():
                df = df.dropna(axis=1, how='all')  ## drop columns with all na values
                # KNN Imputation with k=15
                df_imputed = knn_impute(df)
                # df_imputed = df.fillna(0)
                df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        processed_dataframes[name] = df
    return processed_dataframes

## process train data
def process_traindf(train_dataframes):
    processed_train_dataframes = {}
    stats = {}
    scalers = {}
    # processed_train_dataframes['survival'] = train_dataframes['survival']
    # processed_train_dataframes['clinical'] = train_dataframes['clinical']

    for name, df in train_dataframes.items():
        if name == 'survival':
            continue

        if name == 'clinical':
            scaler = MinMaxScaler()
            tf_data = scaler.fit_transform(df)
            df_new = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
            for col in df_new.columns:
                if df[col].nunique() <= 5:
                    df_new[col] = df[col]
            # df_new.loc[:, "history_other_malignancy__no"] = df.loc[:, "history_other_malignancy__no"]
            scalers[name] = scaler
            processed_train_dataframes[name] = df_new
            continue

        array = df.to_numpy(dtype="float")
        if np.max(array) > 100:
            array = np.log2(array + 1)

        q30 = np.quantile(array, 0.3, axis=0)
        q70 = np.quantile(array, 0.7, axis=0)
        quant = {'q30': q30, 'q70': q70}

        disc_array1 = np.where(array < quant['q30'], -1, 0)
        disc_array2 = np.where(array > quant['q70'], 1, 0)
        disc_array = disc_array1 + disc_array2
        df_new = pd.DataFrame(disc_array, columns=df.columns, index=df.index)
        processed_train_dataframes[name] = df_new
        stats[name] = quant
    return (processed_train_dataframes, stats, scalers)


## Now process val/test dataframes
def process_testdf(test_dataframes, stats, scalers):
    processed_test_dataframes = {}
    # processed_test_dataframes['survival'] = test_dataframes['survival']
    # processed_test_dataframes['clinical'] = test_dataframes['clinical']

    for name, df in test_dataframes.items():
        if name == 'survival':
            continue

        if name == 'clinical':
            tf_data = scalers[name].transform(df)
            df_new = pd.DataFrame(tf_data, columns=df.columns, index=df.index)
            for col in df_new.columns:
                if df[col].nunique() <= 5:
                    df_new[col] = df[col]
            # df_new.loc[:, "history_other_malignancy__no"] = df.loc[:, "history_other_malignancy__no"]
            processed_test_dataframes[name] = df_new
            continue

        array = df.to_numpy(dtype="float")
        if np.max(array) > 100:
            array = np.log2(array + 1)

        if name in stats:
            stat = stats[name]

            disc_array1 = np.where(array < stat['q30'], -1, 0)
            disc_array2 = np.where(array > stat['q70'], 1, 0)
            array = disc_array1 + disc_array2
        df_new = pd.DataFrame(array, columns=df.columns, index=df.index)
        processed_test_dataframes[name] = df_new
    return processed_test_dataframes


## select features using random Forest
def feature_select(df, labels, nFeats):
    clf = RandomForestClassifier(random_state=1234)
    clf.fit(df, labels)
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': df.columns,
        'Importance': importances
    })
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    top_features = feature_importances.head(nFeats)['Feature'].tolist()
    filtered_df = df[top_features]
    return filtered_df


def featselect_df(train_dataframes, labels_df, test_dataframes):
    processed_train_dataframes = {}
    processed_train_dataframes['clinical'] = train_dataframes['clinical']
    processed_test_dataframes = {}
    processed_test_dataframes['clinical'] = test_dataframes['clinical']
    labels = labels_df['status']

    for name, df in train_dataframes.items():
        if name == 'clinical':
            continue

        if name == 'mRNATPM':
            nFeats = 400  ## based on paper
            filtered_df = feature_select(df, labels, nFeats)

        if name == 'cnv':
            nFeats = 200  ## based on paper
            filtered_df = feature_select(df, labels, nFeats)
        processed_train_dataframes[name] = filtered_df
        processed_test_dataframes[name] = test_dataframes[name].loc[:, filtered_df.columns]
    return processed_train_dataframes, processed_test_dataframes


class MDNNMD():
    def __init__(self):
        self.name = 'MDNNMD'
        self.K = 10
        self.alpha = 0.4
        self.beta = 0.1
        self.gamma = 0.5
        self.LABEL = 'os_label_1980'
        self.Kfold = "data/METABRIC_5year_skfold_1980_491.dat"
        self.epsilon = 1e-3
        self.BATCH_SIZE = 64
        self.END_LEARNING_RATE = 0.001
        self.F_SIZE = 400
        self.hidden_units = [3000,3000,3000,100]
        self.MT_CLASS_TASK1 = 2
        self.IS_PT = "F"
        self.MODEL = dict()
        self.IS_PRINT_INFO = "T"
        self.TRAINING = "True"
        self.active_fun = 'tanh'
        self.drop = 0.5
        self.regular = True
        self.lrd = False
        self.curr_fold = 1
        self.MAX_STEPS = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
        self.epoch = 100

    def scale_max_min(self, data, lower=0, upper=1):
        max_value = np.max(np.max(data, 0),0)
        min_value = np.min(np.min(data, 0),0)
        r = np.size(data, 0)
        c = np.size(data, 1)
        for i in range(r):
            for j in range(c):
                data[i,j] = lower + (upper-lower)*((data[i,j]-min_value)/(max_value-min_value))
        return data

    def next_batch(self,train_f,train_l1,batch_size,i):
        # num = int((train_f.shape[0])/batch_size-1)
        num = int(math.floor((train_f.shape[0])/batch_size-1))
        # train_indc = range(train_f.shape[0])

        if train_f.shape[0] > 0:
            train_indc = range(train_f.shape[0])
        else:
            raise ValueError("No training samples available")

        if num == 0:
            random.shuffle(train_indc)
            xs = train_f[train_indc[0:batch_size]]
            y1 = train_l1[train_indc[0:batch_size]]
        else:
            i = i%num
            if i == num-1:
                random.shuffle(train_indc)
            xs = train_f[train_indc[i*batch_size:(i+1)*batch_size]]
            y1 = train_l1[train_indc[i*batch_size:(i+1)*batch_size]]

        return xs,y1

    def batch_norm_wrapper(self,inputs,is_training,decay = 0.999):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))

        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]))

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1])

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([train_mean,train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, self.epsilon)

        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, self.epsilon)

    def code_lables(self, d_class, num_class):
    # #[1,2]  -->  [1,0][0,1]
        coding = []
        cls = []
        labels = np.array(np.zeros(len(d_class)))
        j = -1
        for row in d_class:
            j = j + 1
            labels[j] = row
            for i in range(num_class):
            # for i in [1,7]:
                if row == i:
                    coding.append(1)
                else:
                    coding.append(0)
            cls.append(coding)
            coding = []
        cls = np.array(cls).astype(float)
        return labels, cls

    def packaging_model(self, weight1, biase1, weight2, biase2, weight3, biase3, Y1_weight, Y1_biase, Y1_weight_fc1, Y1_biase_fc1):
        # model = dict()
        # model["weight1"] = weight1
        # model["biase1"] = biase1
        # model["weight2"] = weight2
        # model["biase2"] = biase2
        # model["weight3"] = weight3
        # model["biase3"] = biase3
        # model["Y1_weight"] = Y1_weight
        # model["Y1_biase"] = Y1_biase
        # model["Y1_weight_fc1"] = Y1_weight_fc1
        # model["Y1_biase_fc1"] = Y1_biase_fc1
        # return model

        model = dict()
        # Original weights and biases
        original_weights = {
            "weight1": weight1, "biase1": biase1,
            "weight2": weight2, "biase2": biase2,
            "weight3": weight3, "biase3": biase3,
            "Y1_weight": Y1_weight, "Y1_biase": Y1_biase,
            "Y1_weight_fc1": Y1_weight_fc1, "Y1_biase_fc1": Y1_biase_fc1
        }
        model.update(original_weights)

        # Batch norm parameters
        layers = ['hidden1', 'hidden2', 'hidden3', 'dcl1']
        bn_vars = ['scale', 'beta', 'pop_mean', 'pop_var']
        var_indices = {'scale': 2, 'beta': 3, 'pop_mean': 4, 'pop_var': 5}

        for layer in layers:
            for var_name in bn_vars:
                var_idx = var_indices[var_name]
                var = [v for v in tf.global_variables()
                       if v.name == '{0}/Variable_{1}:0'.format(layer, var_idx)][0]
                model["{0}_{1}".format(layer, var_name)] = var
        return model

    # def train(self, kf1,d_matrix, d_class, cls, ut):
    def train(self, d_matrix, d_class):

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, self.F_SIZE], name='x-input')
            y1_ = tf.placeholder(tf.float32, [None, self.MT_CLASS_TASK1], name='y-input')
            keep_prob = tf.placeholder(tf.float32)
            f_gene_exp = x

        with tf.name_scope('hidden1'):
            weight1 = tf.Variable(tf.truncated_normal([self.F_SIZE, self.hidden_units[0]], stddev=1.0 / math.sqrt(float(self.F_SIZE)/2), seed = 1,name='weights'))
            biase1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[0]]))
            hidden1_mu = tf.matmul(f_gene_exp, weight1) + biase1
            hidden1_BN = self.batch_norm_wrapper(hidden1_mu,  self.TRAINING)

            if self.active_fun == 'relu':
                hidden1 = tf.nn.relu(hidden1_BN )
            else:
                hidden1 = tf.nn.tanh(hidden1_BN )

        with tf.name_scope('hidden2'):
            weight2 = tf.Variable(tf.truncated_normal([self.hidden_units[0], self.hidden_units[1]], stddev=1.0 / math.sqrt(float(self.hidden_units[0])/2),  seed = 1,name='weights'))
            biase2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[1]]))
            hidden2_mu = tf.matmul(hidden1, weight2) + biase2
            hidden2_BN = self.batch_norm_wrapper(hidden2_mu,  self.TRAINING)


            if self.active_fun == 'relu':
                hidden2 = tf.nn.relu(hidden2_BN)
            else:
                hidden2 = tf.nn.tanh(hidden2_BN)

        with tf.name_scope('hidden3'):
            weight3 = tf.Variable(tf.truncated_normal([self.hidden_units[1], self.hidden_units[2]], stddev=1.0 / math.sqrt(float(self.hidden_units[1])/2),  seed = 1,name='weights'))
            biase3 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[2]]))
            hidden3_mu = tf.matmul(hidden2, weight3) + biase3
            hidden3_BN = self.batch_norm_wrapper(hidden3_mu,  self.TRAINING)

            if self.active_fun == 'relu':
                hidden3 = tf.nn.relu(hidden3_BN)
            else:
                hidden3 = tf.nn.tanh(hidden3_BN)

        # # dropout layer
        with tf.name_scope('dcl1'):
            Y1_weight = tf.Variable(tf.truncated_normal([self.hidden_units[2], self.hidden_units[3]], stddev=1.0 / math.sqrt(float(self.hidden_units[2])/2),  seed = 1,name='weights'))
            Y1_biase = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[3]]))
            Y1_h_dc1_mu = tf.matmul(hidden3, Y1_weight) + Y1_biase
            Y1_h_dc1_BN = self.batch_norm_wrapper(Y1_h_dc1_mu,  self.TRAINING)

            if self.active_fun == 'relu':
                Y1_h_dc1_drop  = tf.nn.relu(Y1_h_dc1_BN)
            else:
                Y1_h_dc1_drop  = tf.nn.tanh(Y1_h_dc1_BN)

            Y1_h_dc1_drop_c = tf.nn.dropout(Y1_h_dc1_drop, keep_prob)

        with tf.name_scope('full_connected'):
            Y1_weight_fc1 = tf.Variable(tf.truncated_normal([self.hidden_units[3], self.MT_CLASS_TASK1], stddev=1.0 / math.sqrt(float(self.hidden_units[3])/2), seed = 1, name='weight-Y1-fc'))
            Y1_biase_fc1 = tf.Variable(tf.constant(0.1, shape=[self.MT_CLASS_TASK1]))

            Y1_pre = (tf.matmul(Y1_h_dc1_drop_c, Y1_weight_fc1) + Y1_biase_fc1)
            Y1 = tf.nn.softmax(Y1_pre)


        with tf.name_scope('cross_entropy'):
            l2_loss = 0
            if self.regular:
                l2_loss = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2) + tf.nn.l2_loss(weight3)  + tf.nn.l2_loss(Y1_weight) + tf.nn.l2_loss(Y1_weight_fc1)
                beta = 1e-4
                l2_loss *= beta
            Y1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y1_, logits = Y1_pre))
            Joint_loss = Y1_cross_entropy + l2_loss

            valid_loss = tf.summary.scalar('valid_loss', Joint_loss)
            train_loss = tf.summary.scalar('train_loss', Joint_loss)

        with tf.name_scope('training'):
            if self.lrd:
                cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
                starter_learning_rate = 0.4
                learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(Joint_loss, global_step=cur_step)
            else:
                train_step = tf.train.AdamOptimizer(self.END_LEARNING_RATE).minimize(Joint_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(Y1, 1), tf.argmax(y1_, 1))


            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # def run_train(train_f,train_l,vali_f,vali_l,test_f,test_l,i_k):
        def run_train(train_f, train_l, i_k):

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            sess = tf.InteractiveSession(config=config)
            tf.global_variables_initializer().run()

            # def feed_dict(train,validation,i):
            def feed_dict(train, i):
                if train:
                    batch_size = self.BATCH_SIZE
                    xs, y1 = self.next_batch(train_f,train_l,batch_size,i)
                    k = self.drop
                return {x: xs, y1_: y1, keep_prob: k}


            epochs = -1
            for i in range(0,self.MAX_STEPS[i_k-1]*10+1):
                self.TRAINING = "True"
                # _,loss,Y1_cross_entropy_my= sess.run([train_step,train_loss,Y1_cross_entropy],feed_dict=feed_dict(True,False,i))
                _,loss,Y1_cross_entropy_my= sess.run([train_step,train_loss,Y1_cross_entropy],feed_dict=feed_dict(True,i))

                #print(Y1_cross_entropy_my)

                if i % 10 == 0:
                    epochs = epochs+1
            sess.close()

        for i in range(2):
            print('K fold: %s' % (i))
            X_train = d_matrix
            y_train = d_class

            label11, cls_train = self.code_lables(y_train, self.MT_CLASS_TASK1)
            run_train(X_train, cls_train, i)

        self.MODEL = self.packaging_model(weight1, biase1, weight2, biase2, weight3, biase3, Y1_weight, Y1_biase, Y1_weight_fc1, Y1_biase_fc1)
        print('training is done!')

    def predict(self, df):
        weight1 = self.MODEL["weight1"]
        biase1 = self.MODEL["biase1"]
        weight2 = self.MODEL["weight2"]
        biase2 = self.MODEL["biase2"]
        weight3 = self.MODEL["weight3"]
        biase3 = self.MODEL["biase3"]
        Y1_weight = self.MODEL["Y1_weight"]
        Y1_biase = self.MODEL["Y1_biase"]
        Y1_weight_fc1 = self.MODEL["Y1_weight_fc1"]
        Y1_biase_fc1 = self.MODEL["Y1_biase_fc1"]

        f_gene_exp = tf.convert_to_tensor(df.values, dtype='float32')
        hidden1_mu = tf.matmul(f_gene_exp, weight1) + biase1
        hidden1_BN = self.batch_norm_wrapper(hidden1_mu, False)

        if self.active_fun == 'relu':
            hidden1 = tf.nn.relu(hidden1_BN)
        else:
            hidden1 = tf.nn.tanh(hidden1_BN)

        hidden2_mu = tf.matmul(hidden1, weight2) + biase2
        hidden2_BN = self.batch_norm_wrapper(hidden2_mu, False)

        if self.active_fun == 'relu':
            hidden2 = tf.nn.relu(hidden2_BN)
        else:
            hidden2 = tf.nn.tanh(hidden2_BN)

        hidden3_mu = tf.matmul(hidden2, weight3) + biase3
        hidden3_BN = self.batch_norm_wrapper(hidden3_mu, False)

        if self.active_fun == 'relu':
            hidden3 = tf.nn.relu(hidden3_BN)
        else:
            hidden3 = tf.nn.tanh(hidden3_BN)

        # # dropout layer
        Y1_h_dc1_mu = tf.matmul(hidden3, Y1_weight) + Y1_biase
        Y1_h_dc1_BN = self.batch_norm_wrapper(Y1_h_dc1_mu, False)

        if self.active_fun == 'relu':
            Y1_h_dc1_drop = tf.nn.relu(Y1_h_dc1_BN)
        else:
            Y1_h_dc1_drop = tf.nn.tanh(Y1_h_dc1_BN)

        Y1_h_dc1_drop_c = tf.nn.dropout(Y1_h_dc1_drop, 1.0)

        Y1_pre = (tf.matmul(Y1_h_dc1_drop_c, Y1_weight_fc1) + Y1_biase_fc1)
        Y1 = tf.nn.softmax(Y1_pre)
        return Y1

    def load_txt(self, d_matrix):

        # d_class = np.loadtxt(self.LABEL, delimiter=' ').reshape(-1, 1)
        # d_matrix = np.loadtxt(op, delimiter=' ')

        # d_matrix = d_matrix[:, 0:f_len]
        self.F_SIZE = d_matrix.shape[1]

        # return d_matrix, d_class


def train(input, survival, hidden_units, learning_rate, epoch, drop, batch_size):
    tf.set_random_seed(1234)

    d_matrix = input.to_numpy()
    d_class = survival.loc[:, "status"].to_numpy().reshape(-1, 1)

    # ut = Utils()
    dnn_md = MDNNMD()
    dnn_md.load_txt(d_matrix)
    dnn_md.epoch = epoch
    dnn_md.MAX_STEPS = [dnn_md.epoch, dnn_md.epoch, dnn_md.epoch, dnn_md.epoch, dnn_md.epoch, dnn_md.epoch,
                             dnn_md.epoch, dnn_md.epoch, dnn_md.epoch,
                             dnn_md.epoch]  # 3000,3000,3000,100 MRMR-400  0504
    # dnn_md1.MAX_STEPS = [50,50,50,50,50,50,50,50,50,50]
    dnn_md.hidden_units = hidden_units
    dnn_md.END_LEARNING_RATE = learning_rate
    if batch_size > 0:
        dnn_md.BATCH_SIZE = batch_size
    if drop > 0:
        dnn_md.drop = drop
    dnn_md.IS_PRINT_INFO = "F"
    label1, cls = dnn_md.code_lables(d_class, dnn_md.MT_CLASS_TASK1)
    dnn_md.train(d_matrix, d_class)

    return dnn_md


def predict(dnn_obj, data):
    pred = dnn_obj.predict(data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Define variable mappings
        base_mapping = {
            'weight1': 'hidden1/Variable:0',
            'biase1': 'hidden1/Variable_1:0',
            'weight2': 'hidden2/Variable:0',
            'biase2': 'hidden2/Variable_1:0',
            'weight3': 'hidden3/Variable:0',
            'biase3': 'hidden3/Variable_1:0',
            'Y1_weight': 'dcl1/Variable:0',
            'Y1_biase': 'dcl1/Variable_1:0',
            'Y1_weight_fc1': 'full_connected/Variable:0',
            'Y1_biase_fc1': 'full_connected/Variable_1:0'
        }

        # Add batch norm mappings
        for layer in ['hidden1', 'hidden2', 'hidden3', 'dcl1']:
            for var, idx in [('scale', 2), ('beta', 3), ('pop_mean', 4), ('pop_var', 5)]:
                model_key = '{0}_{1}'.format(layer, var)
                tensor_name = '{0}/Variable_{1}:0'.format(layer, idx)
                base_mapping[model_key] = tensor_name

        # Assign values
        for model_key, var_name in base_mapping.items():
            if model_key in dnn_obj.MODEL:
                var = sess.graph.get_tensor_by_name(var_name)
                sess.run(tf.assign(var, dnn_obj.MODEL[model_key]))

        result = sess.run(pred)[:, 1]
    return result


def find_optimal_weights(probabilities, true_labels):
    weight_steps = np.arange(0, 1.1, 0.1)
    results = []
    for w1, w2, w3 in product(weight_steps, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) > 1e-10:
            continue

        combined_probs = (w1 * probabilities['mRNATPM'] + w2 * probabilities['cnv'] + w3 * probabilities['clinical'])
        auc = roc_auc_score(true_labels, combined_probs)
        results.append({'w1': w1, 'w2': w2, 'w3': w3, 'auc': auc})

    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['auc'].idxmax()]

    return {
        'optimal_weights': {'mRNATPM':best_result['w1'], 'cnv':best_result['w2'], 'clinical':best_result['w3']},
        'max_auc': best_result['auc'],
        'all_results': results_df
    }

