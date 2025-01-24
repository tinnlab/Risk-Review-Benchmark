library(tidyverse)
library(parallel)
library(matrixStats)
library(prioritylasso)
library(survival)
library(glmnet)
library(coefplot)
if (!require("rsample")) {
  install.packages("rsample")
}
library(rsample)

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
# if (!dir.exists(file.path(resPath, "Priority-Lasso"))) {
#   dir.create(file.path(resPath, "Priority-Lasso"))
# }

run <- function(datPath, resPath, timerecPath) {
  # allFiles <- list.files(datPath)
  # allFiles <- strsplit(allFiles, ".rds") %>% do.call(what = c)
  allFiles <- c("TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA", "TCGA-HNSC",
    "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG", "TCGA-LIHC", "TCGA-LUAD",
    "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC", "TCGA-STAD", "TCGA-UCEC")

  ### some functions
  train_predict <- function(DataList_train, DataList_val) {
    survTrain <- DataList_train$survival
    DataList_train_used <- DataList_train[setdiff(names(DataList_train), "survival")]

    survVal <- DataList_val$survival
    DataList_val_used <- DataList_val[setdiff(names(DataList_val), "survival")]

    ## standardization
    statsTrain <- lapply(names(DataList_train_used), function(dataType) {
      dat <- DataList_train_used[[dataType]]
      sds <- colSds(as.matrix(dat))
      sds <- sds[sds != 0]
      means <- colMeans(as.matrix(dat))[names(sds)]
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
    }) %>% `names<-`(names(DataList_val_used))

    # if ("clinical" %in% names(DataList_train_used)){
    #   # keep <- grep("age_at_initial", colnames(DataList_train_used$clinical))
    #   # DataList_train_used$clinical <- as.matrix(DataList_train_used$clinical[, keep]) %>% `colnames<-` (c("age"))
    #   # DataList_val_used$clinical <- as.matrix(DataList_val_used$clinical[, keep]) %>% `colnames<-` (c("age"))
    #   #
    #   # # prioritylasso requires each block has at least 2 columns, so I use both the original
    #   # # normalized "age" values
    #   # keep2 <- grep("age_at_initial", colnames(DataList_train$clinical))
    #   # DataList_train_used$clinical <- cbind(DataList_train_used$clinical, as.matrix(DataList_train$clinical[, keep2]))
    #   # DataList_val_used$clinical <- cbind(DataList_val_used$clinical, as.matrix(DataList_val$clinical[, keep2]))
    #
    #   colkeeps <- c("age_at_initial_pathologic_diagnosis", "history_other_malignancy__no")
    #   DataList_train_used$clinical <- as.matrix(DataList_train_used$clinical[, colkeeps])
    #   DataList_val_used$clinical <- as.matrix(DataList_val_used$clinical[, colkeeps])
    # }

    ## follow the approach by the approach described by Herrmann et al. [2021] to order the input types
    mean_abscoeff <- c()
    for (i in 1:length(DataList_train_used)) {
      res <- try(glmnet::cv.glmnet(as.matrix(DataList_train_used[[i]]), y = Surv(survTrain$time, survTrain$status),
                              nfolds = 3, type.measure = "deviance", family = "cox", alpha = 0))
      if(is(res, "try-error")){
        break
      }
      mean_abscoeff <- c(mean_abscoeff, mean(abs(coefplot::extract.coef(res, "lambda.min")[, 1])))
    }
    if(!is(res, "try-error")){
      inputorder <- order(mean_abscoeff, decreasing=T)
      DataList_train_used <- DataList_train_used[names(DataList_train_used)[inputorder]]
      DataList_val_used <- DataList_val_used[names(DataList_val_used)[inputorder]]
    }

    blockIndices <- rep(seq_along(DataList_train_used), sapply(DataList_train_used, ncol))
    blocks <- lapply(seq_along(DataList_train_used), function(i) which(blockIndices == i)) %>% `names<-`(paste0("block", seq_along(DataList_train_used)))
    DataList_train_used <- do.call(cbind, DataList_train_used)
    # nTypes <- length(blocks)

    ## training
    pri <- try(prioritylasso(Y = Surv(survTrain$time, survTrain$status), X = DataList_train_used, family = "cox",
              type.measure = "deviance", blocks = blocks,
              # max.coef = rep(1, nTypes), block1.penalization = TRUE,
              # lambda.type = "lambda.min",
                        standardize = F, nfolds = 5))
                        # cvoffset = FALSE)

    if (is(pri, "try-error")) {
      predTrain <- rep(NA, nrow(DataList_train$survival))
      predVal <- rep(NA, nrow(DataList_val$survival))
    }else {
      predTrain <- predict(pri, DataList_train_used, type = "link") %>% as.numeric()
      predTrain <- exp(predTrain)

      DataList_val_used <- do.call(cbind, DataList_val_used)
      predVal <- predict(pri, DataList_val_used, type = "link") %>% as.numeric()
      predVal <- exp(predVal)
    }
    predTrain <- as.data.frame(cbind(predTrain, as.matrix(DataList_train$survival))) %>% `rownames<-`(rownames(DataList_train$survival))
    predVal <- as.data.frame(cbind(predVal, as.matrix(DataList_val$survival))) %>% `rownames<-`(rownames(DataList_val$survival))

    return(list(Train = predTrain, Val = predVal))
  }

  ### run the method
  lapply(allFiles, function(file) {
    print(file)

    if (!dir.exists(file.path(resPath, "Priority-Lasso", file))) {
      dir.create(file.path(resPath, "Priority-Lasso", file))
    }
    if (!dir.exists(file.path(timerecPath, "Priority-Lasso", file))) {
      dir.create(file.path(timerecPath, "Priority-Lasso", file))
    }

    DataList <- readRDS(file.path(datPath, paste0(file, ".rds")))
    dataTypes <- c("mRNATPM", "miRNA", "meth450", "cnv", "clinical", "survival")
    dataTypesUsed <- c("mRNATPM", "cnv", "clinical", "survival")
    DataList <- DataList[dataTypes]
    commonSamples <- Reduce(intersect, lapply(DataList, rownames)) %>% unique()
    DataList <- lapply(dataTypesUsed, function(dataType) {
      dat <- DataList[[dataType]][commonSamples,]
      if (!dataType %in% c("clinical", "survival")) {
        dat[is.na(dat)] <- 0
        if (max(dat) > 100) {
          dat <- log2(dat + 1)
        }
      }
      dat
    }) %>% `names<-`(dataTypesUsed)

    survival <- DataList$survival

    ### 5 times of 10 fold cross-validation
    lapply(c(1:5), function(time){
      print(paste0('Running Time: ', time))
      
      if (!dir.exists(file.path(resPath, "Priority-Lasso", file, paste0('Time', time)))) {
        dir.create(file.path(resPath, "Priority-Lasso", file, paste0('Time', time)))
      }
      if (!dir.exists(file.path(timerecPath, "Priority-Lasso", file, paste0('Time', time)))) {
        dir.create(file.path(timerecPath, "Priority-Lasso", file, paste0('Time', time)))
      }
      
      set.seed(time)
      all_folds <- vfold_cv(survival, v = 10, repeats = 1, strata = status)
      all_folds <- lapply(1:10, function(fold) {
        patientIDs <- rownames(survival)[all_folds$splits[[fold]]$in_id]
      })

      lapply(c(1:10), function(fold) {
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
        
        write.csv(Res$Train, file.path(resPath, "Priority-Lasso", file, paste0('Time', time), paste0("Train_Res_", fold, ".csv")), row.names = T)
        write.csv(Res$Val, file.path(resPath, "Priority-Lasso", file, paste0('Time', time), paste0("Val_Res_", fold, ".csv")), row.names = T)
        write.csv(time_df, file.path(timerecPath, "Priority-Lasso", file, paste0('Time', time), paste0("TimeRec_", fold, ".csv")), row.names = T)
      })
    })
    # write.csv(alltimes, file.path(timerecPath, "Priority-Lasso", file, "TimeRec.csv"), row.names=T)
    return(NULL)
  })
}




