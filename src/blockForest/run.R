library(tidyverse)
library(parallel)
library(matrixStats)
library(blockForest)
library(survival)

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
# if (!dir.exists(file.path(resPath, "blockForest"))) {
#   dir.create(file.path(resPath, "blockForest"))
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

    # if ("clinical" %in% names(DataList_train_used)){
    #   # keep1 <- grep("age_at_initial", colnames(DataList_train$clinical))
    #   #
    #   # clindf <- DataList_train$clinical[, -keep1]
    #   # colVars <- colVars(as.matrix(clindf))
    #   # colVars <- sort(colVars, decreasing = T)
    #   # keep2 <- grep(names(colVars)[1], colnames(DataList_train$clinical))
    #   # keep <- c(keep1, keep2)
    #   #
    #   # DataList_train_used$clinical <- as.matrix(DataList_train_used$clinical[, keep]) %>% `colnames<-` (c("age", "V2"))
    #   # DataList_val_used$clinical <- as.matrix(DataList_val_used$clinical[, keep]) %>% `colnames<-` (c("age", "V2"))
    #
    #   colkeeps <- c("age_at_initial_pathologic_diagnosis", "history_other_malignancy__no")
    #   DataList_train_used$clinical <- as.matrix(DataList_train_used$clinical[, colkeeps])
    #   DataList_val_used$clinical <- as.matrix(DataList_val_used$clinical[, colkeeps])
    # }

    blockIndices <- rep(seq_along(DataList_train_used), sapply(DataList_train_used, ncol))
    blocks <- lapply(seq_along(DataList_train_used), function(i) which(blockIndices == i)) %>% `names<-`(paste0("block", seq_along(DataList_train_used)))
    DataList_train_used <- do.call(cbind, DataList_train_used)
    colnames(DataList_train_used) <- paste0('Var', c(1:ncol(DataList_train_used)))
    train_label <- Surv(survTrain$time, survTrain$status)

    ## training
    set.seed(1234)
    bf <- blockfor(DataList_train_used, train_label, block = blocks, block.method = "BlockForest", num.trees = 2000, replace = TRUE,
              nsets = 300, num.trees.pre = 1500, splitrule = "extratrees", num.threads=1)
    if (is(bf, "try-error")) {
      predTrain <- rep(NA, nrow(DataList_train$survival))
      predVal <- rep(NA, nrow(DataList_val$survival))
    }else {
      predTrain <- predict(bf$forest, data = DataList_train_used, block.method = "BlockForest")
      survFTrain <- predTrain$survival %>% `rownames<-` (rownames(DataList_train_used))
      colnames(survFTrain) <- paste0(rep("predTrain", ncol(survFTrain)), "_", predTrain$unique.death.times)
      predTrain <- survFTrain

      DataList_val_used <- do.call(cbind, DataList_val_used)
      colnames(DataList_val_used) <- paste0('Var', c(1:ncol(DataList_val_used)))

      predVal <- predict(bf$forest, data = DataList_val_used, block.method = "BlockForest")
      survFVal <- predVal$survival %>% `rownames<-` (rownames(DataList_val_used))
      colnames(survFVal) <- paste0(rep("predTrain", ncol(survFVal)), "_", predVal$unique.death.times)
      predVal <- survFVal
    }
    predTrain <- as.data.frame(cbind(predTrain, as.matrix(DataList_train$survival))) %>% `rownames<-`(rownames(DataList_train$survival))
    predVal <- as.data.frame(cbind(predVal, as.matrix(DataList_val$survival))) %>% `rownames<-`(rownames(DataList_val$survival))

    return(list(Train = predTrain, Val = predVal))
  }

  ### run the method
  lapply(allFiles, function(file) {
    print(file)

    if (!dir.exists(file.path(resPath, "blockForest", file))) {
      dir.create(file.path(resPath, "blockForest", file))
    }
    if (!dir.exists(file.path(timerecPath, "blockForest", file))) {
      dir.create(file.path(timerecPath, "blockForest", file))
    }

    DataList <- readRDS(file.path(datPath, paste0(file, ".rds")))
    dataTypes <- c("mRNATPM", "miRNA", "meth450", "cnv", "clinical", "survival")
    dataTypesUsed <- c("mRNATPM", "miRNA", "cnv", "clinical", "survival")
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

      if (!dir.exists(file.path(resPath, "blockForest", file, paste0('Time', time)))) {
        dir.create(file.path(resPath, "blockForest", file, paste0('Time', time)))
      }
      if (!dir.exists(file.path(timerecPath, "blockForest", file, paste0('Time', time)))) {
        dir.create(file.path(timerecPath, "blockForest", file, paste0('Time', time)))
      }

      set.seed(time)
      all_folds <- vfold_cv(survival, v = 10, repeats = 1, strata = status)
      all_folds <- lapply(1:10, function(fold) {
        patientIDs <- rownames(survival)[all_folds$splits[[fold]]$in_id]
      })

      mclapply(1:10, mc.cores=2, function(fold) {
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

        write.csv(Res$Train, file.path(resPath, "blockForest", file, paste0('Time', time), paste0("Train_Res_", fold, ".csv")), row.names = T)
        write.csv(Res$Val, file.path(resPath, "blockForest", file, paste0('Time', time), paste0("Val_Res_", fold, ".csv")), row.names = T)
        write.csv(time_df, file.path(timerecPath, "blockForest", file, paste0('Time', time), paste0("TimeRec_", fold, ".csv")), row.names = T)
      })
    })
    # write.csv(alltimes, file.path(timerecPath, "blockForest", file, "TimeRec.csv"), row.names=T)
    return(NULL)
  })
}


