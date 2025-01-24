# Installing Tensorflow
if (!require("tensorflow")) {
  remotes::install_github("rstudio/tensorflow", upgrade = 'never')
}
# library(tensorflow)
# install_tensorflow(method = 'conda', envname = 'rp-review-env1')
# # Installing keras
if (!require("keras")) {
  install.packages("keras", repos='http://cran.us.r-project.org', dependencies=TRUE)
}
# library(keras)
# install_keras()
# install compound.Cox
if (!require("compound.Cox")) {
  install.packages('compound.Cox', repos='http://cran.us.r-project.org', dependencies=TRUE)
}
if (!require("rsample")) {
  install.packages("rsample")
}
library(rsample)


library(tidyverse)
library(parallel)
library(reticulate)
library(compound.Cox)
library(matrixStats)
library(glmnet)
library(keras)
library(tensorflow)
use_condaenv('~/miniconda3/envs/rp-review-env4')

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)
Sys.setenv(OMP_NUM_THREADS = 1,
           OPENBLAS_NUM_THREADS = 1,
           MKL_NUM_THREADS = 1,
           VECLIB_MAXIMUM_THREADS = 1,
           NUMEXPR_NUM_THREADS = 1)

# ### specify the path to the data and saved results
# datPath <- "/nfs/blanche/share/daotran/SurvivalPrediction/AllData/RP_Data_rds_meth25k"
# resPath <- "/data/dungp/projects_catalina/risk-review/benchmark/run-results"
# if (!dir.exists(file.path(resPath, "Multimodal_NSCLC"))) {
#   dir.create(file.path(resPath, "Multimodal_NSCLC"))
# }

# IslandProbes <- readRDS("/nfs/blanche/share/daotran/SurvivalPrediction/AllData/Meth_Island_Probe.rds")$ProbeIds
# LncRNAList <- readRDS("/nfs/blanche/share/daotran/SurvivalPrediction/AllData/LncRNAs.rds")$LNCS

run <- function(datPath, resPath, timerecPath, IslandProbes, LncRNAList) {
  allFiles <- c("TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA", "TCGA-HNSC",
                "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG", "TCGA-LIHC", "TCGA-LUAD",
                "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC", "TCGA-STAD", "TCGA-UCEC")

  ## define functions
  linear_featureselection_func <- function(traindata, testdata, time, status, num_features) {
    if (ncol(traindata) <= num_features){
      return(list(traindata, testdata))
    }
    set.seed(1234)
    associat <- uni.selection(t.vec = time, d.vec = status, X.mat = traindata,
                              P.value = 0.8, randomize = TRUE, K = 5)
    col_filtered <- names(associat$P)[1:num_features]
    col_filtered <- col_filtered[!is.na(col_filtered)]

    if (length(col_filtered) == 0){
      return(list(traindata, testdata))
    }

    traindata <- traindata[, col_filtered]
    testdata <- testdata[, col_filtered]

    lfs_list <- list(traindata, testdata)
    return(lfs_list)
  }


  denoising_zeros_func <- function(train_data, test_data, num_features, zeros_percentage, activation_function) {
    set.seed(1234)

    corrupt_with_ones <- function(x) {
      n_to_sample <- floor(length(x) * zeros_percentage)
      elements_to_corrupt <- sample(seq_along(x), n_to_sample, replace = FALSE)
      x[elements_to_corrupt] <- 0
      return(x)
    }

    early_stop <- keras::callback_early_stopping(monitor = "val_loss", min_delta = 0.001,
                                                patience = 5, restore_best_weights = TRUE, verbose = 1)

    inputs_currupted_ones <- train_data %>%
      as.data.frame() %>%
      purrr::map_df(corrupt_with_ones)
    features <- as.matrix(train_data)
    inputs_currupted_ones <- as.matrix(inputs_currupted_ones)
    test_data <- as.matrix(test_data)

    model1 <- keras_model_sequential()
    model1 %>%
      layer_dense(units = num_features, activation = activation_function, input_shape = c(ncol(inputs_currupted_ones)), name = "BottleNeck") %>%
      layer_dense(units = ncol(inputs_currupted_ones))

    model1 %>% keras::compile(
      loss = "mean_squared_error",
      optimizer = optimizer_adam(lr = 0.001))

    model1 %>% keras::fit(
      x = inputs_currupted_ones, y = features,
      epochs = 100, validation_split = 0.1,
      callbacks = list(early_stop))

    intermediate_layer_model1 <- keras_model(inputs = model1$input, outputs = get_layer(model1, "BottleNeck")$output)
    denoising_zeros_list <- list(predict(intermediate_layer_model1, inputs_currupted_ones),
                                predict(intermediate_layer_model1, test_data))
    return(denoising_zeros_list)
  }

  train_predict <- function(DataList_train, DataList_val) {
    survTrain <- DataList_train$survival
    DataList_train_used <- DataList_train[setdiff(names(DataList_train), "survival")]

    survVal <- DataList_val$survival
    DataList_val_used <- DataList_val[setdiff(names(DataList_val), "survival")]

    ## standardization
    statsTrain <- lapply(names(DataList_train_used), function(dataType) {
      dat <- DataList_train_used[[dataType]]
      sds <- colSds(as.matrix(dat), na.rm=T)
      sds <- sds[sds != 0]
      means <- colMeans(as.matrix(dat), na.rm=T)[names(sds)]
      if (dataType == "clinical") {
        cateid <- c()
        for (col in names(means)) {
          if (length(unique(dat[, col])) <= 5) {
            cateid <- c(cateid, col)
          }
        }
        means[cateid] <- 0
        sds[cateid] <- 1
      }
      list(means = means, sds = sds)
    }) %>% `names<-`(names(DataList_train_used))

    DataList_train_used <- lapply(names(DataList_train_used), function(dataType) {
      dat <- DataList_train_used[[dataType]]
      means <- statsTrain[[dataType]]$means
      sds <- statsTrain[[dataType]]$sds
      dat <- as.matrix(dat[, names(means)])
      dat <- sweep(dat, 2, means, `-`)
      dat <- sweep(dat, 2, sds, `/`)

      replace_na_with_median <- function(df) {
        df %>% mutate(across(where(~any(is.na(.))), ~replace_na(., median(., na.rm = TRUE))))
      }

      dat <- replace_na_with_median(as.data.frame(dat))
      as.matrix(dat)
      # dat <- sweep(dat, 2, sds, `/`)[, c(1:10)]
    }) %>% `names<-`(names(DataList_train_used))

    DataList_val_used <- lapply(names(DataList_val_used), function(dataType) {
      dat <- DataList_val_used[[dataType]]
      means <- statsTrain[[dataType]]$means
      sds <- statsTrain[[dataType]]$sds
      dat <- as.matrix(dat[, names(means)])
      dat <- sweep(dat, 2, means, `-`)
      dat <- sweep(dat, 2, sds, `/`)
      # dat <- sweep(dat, 2, sds, `/`)[, c(1:10)]
      replace_na_with_median <- function(df) {
        df %>% mutate(across(where(~any(is.na(.))), ~replace_na(., median(., na.rm = TRUE))))
      }

      dat <- replace_na_with_median(as.data.frame(dat))
      as.matrix(dat)
    }) %>% `names<-`(names(DataList_val_used))

    # if ("clinical" %in% names(DataList_train_used)) {
      # keep <- grep("age_at_initial", colnames(DataList_train_used$clinical))
      # clinTrain <- as.matrix(DataList_train_used$clinical[, keep]) %>% `colnames<-`(c("age"))
      # DataList_train_used <- DataList_train_used[setdiff(names(DataList_train_used), "clinical")]
      # clinVal <- as.matrix(DataList_val_used$clinical[, keep]) %>% `colnames<-`(c("age"))
      # DataList_val_used <- DataList_val_used[setdiff(names(DataList_val_used), "clinical")]

      # colkeeps <- c("age_at_initial_pathologic_diagnosis", "history_other_malignancy__no")
      # clinTrain <- as.matrix(DataList_train_used$clinical[, colkeeps])
      # clinVal <- as.matrix(DataList_val_used$clinical[, colkeeps])
      # DataList_train_used <- DataList_train_used[setdiff(names(DataList_train_used), "clinical")]
      # DataList_val_used <- DataList_val_used[setdiff(names(DataList_val_used), "clinical")]
    # }

    clinTrain <- as.matrix(DataList_train_used$clinical)
    clinVal <- as.matrix(DataList_val_used$clinical)
    DataList_train_used <- DataList_train_used[setdiff(names(DataList_train_used), "clinical")]
    DataList_val_used <- DataList_val_used[setdiff(names(DataList_val_used), "clinical")]

    ### begin training
    # set hyperparameters
    activation_function_set <- c("sigmoid") #,"tanh","relu"
    zeros_percentage_set <- c(0.3) #, 0.2, 0.4)
    u <- 1
    z <- 1

    # The number of features the linear feature selection technique will choose for each modality
    nFeaturesSe <- list(mRNATPM = 500, miRNA = 300, meth450 = 500, lncRNA = 300)
    # The number of features the denoising autoencoder will reduce the feature space to
    nDimsNew <- list(mRNATPM = 50, miRNA = 30, meth450 = 50, lncRNA = 30)

    zeros_percentage <- zeros_percentage_set[z]
    activation_function <- activation_function_set[u]

    AEresults <- lapply(names(DataList_train_used), function(dataType) {
      traindata <- DataList_train_used[[dataType]]
      testdata <- DataList_val_used[[dataType]]
      nFeat <- nFeaturesSe[[dataType]]
      nDim <- nDimsNew[[dataType]]
      newData_list <- linear_featureselection_func(traindata, testdata, survTrain$time, survTrain$status, nFeat)
      denoising_zeros_list <- denoising_zeros_func(newData_list[[1]], newData_list[[2]], nDim, zeros_percentage, activation_function)
    }) %>% `names<-`(names(DataList_train_used))

    train_data <- cbind(do.call(cbind, lapply(AEresults, `[[`, 1)), clinTrain) %>% `rownames<-` (rownames(clinTrain))
    test_data <- cbind(do.call(cbind, lapply(AEresults, `[[`, 2)), clinVal) %>% `rownames<-` (rownames(clinVal))

    mm_nsclc <- try({ cv.glmnet(train_data, Surv(survTrain$time, survTrain$status), alpha = 0.5, # lasso: alpha = 1; ridge: alpha=0
                      family = "cox", type.measure = "C") })

    if (is(mm_nsclc, "try-error")) {
      predTrain <- rep(NA, nrow(DataList_train$survival))
      predVal <- rep(NA, nrow(DataList_val$survival))
    }else {
      predTrain <- predict(mm_nsclc, train_data, type = "link")
      predTrain <- exp(predTrain)

      predVal <- predict(mm_nsclc, test_data, type = "link")
      predVal <- exp(predVal)
    }
    predTrain <- as.data.frame(cbind(predTrain, as.matrix(survTrain))) %>% `rownames<-`(rownames(survTrain))
    colnames(predTrain)[1] <- "predTrain"
    predVal <- as.data.frame(cbind(predVal, as.matrix(survVal))) %>% `rownames<-`(rownames(survVal))
    colnames(predVal)[1] <- "predVal"
    return(list(Train = predTrain, Val = predVal))
  }

  lapply(allFiles, function(file) {
    print(file)

    if (!dir.exists(file.path(resPath, "Multimodal_NSCLC", file))) {
      dir.create(file.path(resPath, "Multimodal_NSCLC", file))
    }
    if (!dir.exists(file.path(timerecPath, "Multimodal_NSCLC", file))) {
      dir.create(file.path(timerecPath, "Multimodal_NSCLC", file))
    }

    DataList <- readRDS(file.path(datPath, paste0(file, ".rds")))
    dataTypes <- c("mRNATPM", "miRNA", "cnv", "meth450", "clinical", "survival")
    dataTypesUsed <- c("mRNATPM", "miRNA", "meth450", "clinical", "survival", "lncRNA")
    DataList <- DataList[dataTypes]

    ## generate lncRNA data
    lncRNA <- DataList$mRNATPM
    All_lRNA <- strsplit(colnames(lncRNA), "___")
    All_lRNA <- lapply(All_lRNA, function(i) { i[[1]] }) %>% do.call(what = c)
    lncRNA <- lncRNA[, All_lRNA %in% LncRNAList]
    DataList$lncRNA <- lncRNA

    survival <- DataList$survival
    survival <- survival[survival$time > 1,]
    DataList$survival <- survival

    commonSamples <- Reduce(intersect, lapply(DataList, rownames)) %>% unique()
    DataList <- lapply(dataTypesUsed, function(dataType) {
      df <- DataList[[dataType]][commonSamples,]
      if (dataType %in% c("survival", "clinical")) {
        return(df)
      }
      if (dataType %in% c("mRNATPM", "miRNA", "lncRNA")) {
        # Remove columns with high missing data/ high number of zero
        df <- df[, colMeans(is.na(df) | df == 0) <= 0.2]
        if (max(df, na.rm=T) > 100) {
          df <- log2(df + 1)
        }
      }
      if (dataType == "meth450") {
        df <- df[, colnames(df) %in% IslandProbes]
        df <- df[, colMeans(is.na(df)) <= 0.2]
        ranks <- colVars(as.matrix(df), na.rm = TRUE)
        ranks <- sort(ranks, decreasing = TRUE)
        df <- df[, names(ranks)[1:min(c(25000, ncol(df)))]] #top 25,000 methylation probes by variance
      }
      df
    }) %>% `names<-`(dataTypesUsed)

    survival <- DataList$survival

    ### 5 times of 10 fold cross-validation
    lapply(c(1:5), function(time){
      # if (time != 3){
      #   return(NULL)
      # }
      print(paste0('Running Time: ', time))

      if (!dir.exists(file.path(resPath, "Multimodal_NSCLC", file, paste0('Time', time)))) {
        dir.create(file.path(resPath, "Multimodal_NSCLC", file, paste0('Time', time)))
      }
      if (!dir.exists(file.path(timerecPath, "Multimodal_NSCLC", file, paste0('Time', time)))) {
        dir.create(file.path(timerecPath, "Multimodal_NSCLC", file, paste0('Time', time)))
      }

      set.seed(time)
      all_folds <- vfold_cv(survival, v = 10, repeats = 1, strata = status)
      all_folds <- lapply(1:10, function(fold) {
        patientIDs <- rownames(survival)[all_folds$splits[[fold]]$in_id]
      })

      lapply(c(1:10), function(fold) {
        # if (fold != 10){
        #   return(NULL)
        # }
        k_clear_session()
        print(paste0('Running Fold: ', fold))
        trainIndex <- all_folds[[fold]]
        valIndex <- setdiff(rownames(survival), trainIndex)

        DataList_train <- lapply(DataList, function(x) x[trainIndex,])
        DataList_val <- lapply(DataList, function(x) x[valIndex,])

        start_time <- Sys.time()
        Res <- train_predict(DataList_train, DataList_val)
        end_time <- Sys.time()
        record_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
        print(paste0("running time: ", record_time))

        time_df <- data.frame(
          dataset = file,
          time_point = time,
          fold = fold,
          runtime_seconds = record_time
        )

        write.csv(Res$Train, file.path(resPath, "Multimodal_NSCLC", file, paste0('Time', time), paste0("Train_Res_", fold, ".csv")), row.names = T)
        write.csv(Res$Val, file.path(resPath, "Multimodal_NSCLC", file, paste0('Time', time), paste0("Val_Res_", fold, ".csv")), row.names = T)
        write.csv(time_df, file.path(timerecPath, "Multimodal_NSCLC", file, paste0('Time', time), paste0("TimeRec_", fold, ".csv")), row.names = T)
      })
    })
    # write.csv(alltimes, file.path(timerecPath, "Multimodal_NSCLC", file, "TimeRec.csv"), row.names=T)
    return(NULL)
  })
}


