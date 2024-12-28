import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import torch


## intersect genes
def intersect_columns(df1, df2, df3, table):
    common_columns = list(set(df1.columns) & set(df2.columns) & set(df3.columns))
    unique_genes = set(table['GeneSymbol'].dropna().unique())
    intersected_columns = [col for col in common_columns if col in unique_genes]
    return intersected_columns


## knn impute function
def knn_impute(X, k=1):
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
def impute_df(dataframes, genes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        # if name == "clinical":
        #     filtered_columns = ["age_at_initial_pathologic_diagnosis", "history_other_malignancy__no"]
        #     df = df.loc[:, filtered_columns]

        df = df[genes] ## check it with a small number of genes first, remove the indices later
        if df.max().max(skipna=True) > 100:
            df = np.log2(df + 1)
        if df.isna().any().any():
            df = df.dropna(axis=1, how='all') ## drop columns with all na values
            # KNN Imputation with k=1
            df_imputed = knn_impute(df)
            # df_imputed = df.fillna(0)   ### change back later
            df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
        processed_dataframes[name] = df
    return processed_dataframes


## process train dataframes
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
        processed_train_dataframes[name] = df_new
    return (processed_train_dataframes, scalers)


## Now process test dataframes
def process_testdf(test_dataframes, scalers):
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

        processed_test_dataframes[name] = df_new
    return processed_test_dataframes


## generate the omics-combined data for training
def gene_omics_train(dataframes):
    mRNA = dataframes['mRNATPM_map']
    cnv = dataframes['cnv_map']
    meth450 = dataframes['meth450_map']

    def generate_combined_dataframe(mRNA, cnv, meth):
        combined_data = {}

        for gene in mRNA.columns:
            combined_data[gene +'_mRNA'] = mRNA[gene]
            combined_data[gene +'_cnv'] = cnv[gene]
            combined_data[gene +'_meth'] = meth[gene]
            combined_data[gene + '_mRNA_cnv'] = mRNA[gene] * cnv[gene]
            combined_data[gene + '_mRNA_meth'] = mRNA[gene] * meth[gene]

        combined_df = pd.DataFrame(combined_data)
        return combined_df

    df = generate_combined_dataframe(mRNA, cnv, meth450)
    return df


## load data function
def load_data(omics, clinical, survival):
    survival.sort_values("time", ascending=False, inplace=True)
    # clinical = clinical.loc[survival.index, :]
    omics = omics.loc[survival.index, :].values
    ytime = survival.loc[:, ["time"]].values
    yevent = survival.loc[:, ["status"]].values
    age = clinical.loc[survival.index, ["age_at_initial_pathologic_diagnosis"]].values

    x = torch.from_numpy(omics).to(dtype=torch.float).cuda()
    ytime = torch.from_numpy(ytime).to(dtype=torch.float).cuda()
    yevent = torch.from_numpy(yevent).to(dtype=torch.float).cuda()
    age = torch.from_numpy(age).to(dtype=torch.float).cuda()

    return (x, ytime, yevent, age)


## generate data for prediction
def pred_data_gene(dataframes):
    omics = gene_omics_train(dataframes)
    clin = dataframes['clinical']
    surv = dataframes['survival']

    omics = omics.values
    age = clin.loc[:, ["age_at_initial_pathologic_diagnosis"]].values

    x_pred = torch.from_numpy(omics).to(dtype=torch.float).cuda()
    age_pred = torch.from_numpy(age).to(dtype=torch.float).cuda()
    return(x_pred, age_pred, surv)


## generate gene indices and pathway indices arrays
def generate_id_array(combined_df, pathway_df):
    column_names = combined_df.columns
    unique_genes = []
    gene_to_id = {}
    for name in column_names:
        gene = name.split('_')[0]
        if gene not in gene_to_id:
            gene_to_id[gene] = len(unique_genes)
            unique_genes.append(gene)

    gene_id_array = np.zeros((2, len(column_names)), dtype=int)
    for col_idx, col_name in enumerate(column_names):
        gene = col_name.split('_')[0]
        gene_id_array[0, col_idx] = gene_to_id[gene]
    gene_id_array[1, :] = np.arange(len(column_names))

    filtered_pw_df = pathway_df[
        pathway_df['GeneSymbol'].isin(gene_to_id.keys())
    ]
    unique_pathways = list(dict.fromkeys(filtered_pw_df['Pathway']))
    pathway_to_id = {pathway: idx for idx, pathway in enumerate(unique_pathways)}

    pathway_indices_list = [pathway_to_id[row['Pathway']] for _, row in filtered_pw_df.iterrows()]
    gene_indices_list = [gene_to_id[row['GeneSymbol']] for _, row in filtered_pw_df.iterrows()]
    pw_id_array = np.array([pathway_indices_list, gene_indices_list])

    return gene_id_array, pw_id_array


