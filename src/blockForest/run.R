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

run <- function(datPath, resPath) {
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
    train_label <- Surv(survTrain$time, survTrain$status)

    ## training
    set.seed(1234)
    bf <- blockfor(DataList_train_used, train_label, block = blocks, block.method = "BlockForest", num.trees = 200, replace = TRUE,
                nsets = 30, num.trees.pre = 150, splitrule = "extratrees", num.threads=1)
    if (is(bf, "try-error")) {
      predTrain <- rep(NA, nrow(DataList_train$survival))
      predVal <- rep(NA, nrow(DataList_val$survival))
    }else {
      predTrain <- predict(bf$forest, data = DataList_train_used, block.method = "BlockForest")
      survFTrain <- predTrain$survival %>% `rownames<-` (rownames(DataList_train_used))
      colnames(survFTrain) <- paste0(rep("predTrain", ncol(survFTrain)), "_", predTrain$unique.death.times)
      predTrain <- survFTrain

      DataList_val_used <- do.call(cbind, DataList_val_used)
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
  mclapply(allFiles, mc.cores = 10, function(file) {
    print(file)

    if (!dir.exists(file.path(resPath, "blockForest", file))) {
      dir.create(file.path(resPath, "blockForest", file))
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
    nSamples <- nrow(survival)
    aliveIndex <- which(survival$status == 0)
    deadIndex <- which(survival$status == 1)

    lapply(1:10, function(seed) {
      print(seed)
      set.seed(seed)
      trainIndex <- c(sample(aliveIndex, size = floor(length(aliveIndex) * 0.8)), sample(deadIndex, size = floor(length(deadIndex) * 0.8)))
      valIndex <- setdiff(seq_len(nSamples), trainIndex)

      DataList_train <- lapply(DataList, function(x) x[trainIndex,])
      DataList_val <- lapply(DataList, function(x) x[valIndex,])

      Res <- train_predict(DataList_train, DataList_val)
      write.table(Res$Train, file.path(resPath, "blockForest", file, paste0("Train_Res_", seed, ".csv")), col.names = T, sep = ",")
      write.table(Res$Val, file.path(resPath, "blockForest", file, paste0("Val_Res_", seed, ".csv")), col.names = T, sep = ",")
      return(NULL)
    })
    return(NULL)
  })
}


