import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

## filter function
def filter_df(dataframes):
    processed_dataframes = {}
    processed_dataframes['survival'] = dataframes['survival']
    processed_dataframes['clinical'] = dataframes['clinical']

    for name, df in dataframes.items():
        if name == 'survival' or name == 'clinical':
            continue

        else:
            df = df.fillna(0)
            if df.max().max() > 100:
                df = np.log2(df + 1)
        processed_dataframes[name] = df
    return processed_dataframes

## run lmqcm in python
def runlmqcm(trainDat, testDat, dataType):
    ro.r('library(lmQCM)')
    ro.r('library(dplyr)')

    ro.globalenv['trainDat'] = pandas2ri.py2rpy(trainDat)
    ro.globalenv['testDat'] = pandas2ri.py2rpy(testDat)

    if dataType == 'mRNATPM':
        # Define and execute the R code
        r_code = """
        lmQCM_res <- lmQCM(t(trainDat), gamma=0.7, lambda=1, t=1, beta=0.4, minClusterSize=10)
        clusters <- lmQCM_res@clusters.names
        eigenMat <- lmQCM_res@eigengene.matrix
        trainDat <- as.data.frame(t(eigenMat))
    
        # Select genes for test data
        geneCluster <- lapply(clusters, function(cluster){
          gene <- cluster
        }) %>% do.call(what = c)
        geneCluster <- unique(geneCluster)
        testDat <- testDat[, geneCluster]
    
        # Process test data using clusters
        testDat <- lapply(clusters, function(cluster){
          Mat <- as.matrix(testDat[, cluster])
          svdRes <- svd(Mat)
          vec <- svdRes$u[, 1]
        }) %>% do.call(what = cbind) %>% as.data.frame() %>% `rownames<-` (rownames(testDat))
        """

    if dataType == 'miRNA':
        r_code = """
                lmQCM_res <- lmQCM(t(trainDat), gamma=0.4, lambda=1, t=1, beta=0.6, minClusterSize=4)
                clusters <- lmQCM_res@clusters.names
                eigenMat <- lmQCM_res@eigengene.matrix
                trainDat <- as.data.frame(t(eigenMat))

                # Select genes for test data
                geneCluster <- lapply(clusters, function(cluster){
                  gene <- cluster
                }) %>% do.call(what = c)
                geneCluster <- unique(geneCluster)
                testDat <- testDat[, geneCluster]

                # Process test data using clusters
                testDat <- lapply(clusters, function(cluster){
                  Mat <- as.matrix(testDat[, cluster])
                  svdRes <- svd(Mat)
                  vec <- svdRes$u[, 1]
                }) %>% do.call(what = cbind) %>% as.data.frame() %>% `rownames<-` (rownames(testDat))
                """
    ro.r(r_code)

    # Retrieve processed trainDat and testDat
    trainDat_processed = pandas2ri.rpy2py(ro.globalenv['trainDat'])
    testDat_processed = pandas2ri.rpy2py(ro.globalenv['testDat'])

    return trainDat_processed, testDat_processed

# process train and test data
def process_df(train_dataframes, test_dataframes):
    processed_train_dataframes = {}
    # processed_train_dataframes['survival'] = train_dataframes['survival']
    processed_train_dataframes['clinical'] = train_dataframes['clinical']

    processed_test_dataframes = {}
    # processed_test_dataframes['survival'] = test_dataframes['survival']
    processed_test_dataframes['clinical'] = test_dataframes['clinical']

    for name, df_train in train_dataframes.items():
        df_test = test_dataframes[name]
        if name == 'survival' or name == 'clinical':
            continue

        if name == 'mRNATPM' or name == 'miRNA':
            gene_means = df_train.mean(axis=0)
            mean_threshold = gene_means.quantile(0.2)
            gene_variances = df_train.var(axis=0)
            variance_threshold = gene_variances.quantile(0.2)
            df_train_new = df_train.loc[:, (gene_means > mean_threshold) & (gene_variances > variance_threshold)]
            # df_test_new = df_test.loc[:, df_train.columns]

            ## run the lmQCM
            # df_train_new = fast_filter(df_train_new.T, 0.6, 0.6)
            # df_train_new = df_train_new.T
            df_test_new = df_test.loc[:, df_train_new.columns]
            df_train_new, df_test_new = runlmqcm(df_train_new, df_test_new, name)

        if name == 'cnv':
            row_sums_train = df_train.sum(axis=1)
            if row_sums_train.max() > 100:
                row_sums_train = np.log2(row_sums_train + 1)
            df_train_new = pd.DataFrame(row_sums_train, columns=['Row_Sums'])

            row_sums_test = df_test.sum(axis=1)
            if row_sums_test.max() > 100:
                row_sums_test = np.log2(row_sums_test + 1)
            df_test_new = pd.DataFrame(row_sums_test, columns=['Row_Sums'])

        processed_train_dataframes[name] = df_train_new
        processed_test_dataframes[name] = df_test_new

    return processed_train_dataframes, processed_test_dataframes