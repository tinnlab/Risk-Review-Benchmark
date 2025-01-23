# Installing M2EFM
if (!require("M2EFM")) {
  remotes::install_github("jeffreyat/M2EFM")
}
library(M2EFM)
library(tidyverse)
library(parallel)
if (!require("rsample")) {
  install.packages("rsample")
}
library(rsample)
library(impute)

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
# if(!dir.exists(file.path(resPath, "M2EFM"))){
#   dir.create(file.path(resPath, "M2EFM"))
# }

run <- function(datPath, Probes_remove, resPath, timerecPath) {
  # allFiles <- list.files(datPath)
  # allFiles <- strsplit(allFiles, ".rds") %>% do.call(what = c)
  allFiles <- c("TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-COAD", "TCGA-ESCA", "TCGA-HNSC",
    "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG", "TCGA-LIHC", "TCGA-LUAD",
    "TCGA-LUSC", "TCGA-PAAD", "TCGA-SARC", "TCGA-STAD", "TCGA-UCEC")
  # allFiles <- c("TCGA-ESCA", "TCGA-LAML", "TCGA-SARC", "TCGA-STAD", "TCGA-UCEC")



  ### some functions
  impute_meth <- function(df, na_max_prop=.5) {
    load_it("lumi")
    df <- t(df)
    # Filter probes more than half NAs
    df <- df[rowMeans(is.na(df)) <= na_max_prop,]
    #df = beta2m(df)
    df <- data.frame(t(impute.knn(as.matrix(df), k=10, rng.seed=3301)$data))
    return(df)
  }

  train_predict <- function(DataList_train, DataList_val){
    exp <- as.data.frame(t(DataList_train$mRNATPM_map))
    meth <- as.data.frame(t(DataList_train$meth450))

    # Select the distant pairs of genes and probes with significant association
    m2e <- get_m2eqtls(probe_list = system.file("extdata", "PROBELIST.txt", package = "M2EFM"),
                    meth_data = meth,
                    exp_data = exp,
                    num_trans = 110,
                    num_probes = 550,
                    treatment_col = "")

    # Filter the exp + meth data and transform beta values of meth data into m values
    genes <- get_genes(m2e, "trans")
    exp <- exp[genes,]
    probes <- get_probes(m2e)
    meth <- beta2m(meth[probes,])
    meth[is.na(meth)] <- 0
    comb <- merge(t(exp), t(meth), by = c("row.names"))
    rownames(comb) <- comb[, 1]
    comb <- comb[, -1]
    clin <- DataList_train$clinical
    # colIds <- grep("age_at_initial", colnames(clin))
    # colIds <- c("age_at_initial_pathologic_diagnosis", "history_other_malignancy__no")
    colnames(clin) <- paste0(rep("Var", ncol(clin)), c(1:ncol(clin)))

    comb <- comb[rownames(clin), ]

    surv <- DataList_train$survival
    surv <- Surv(surv$time, surv$status)

    m2efm <- try({ build_m2efm(comb, surv, clin, colnames(clin), standardize = FALSE) })
    if (is(m2efm, "try-error")){
      predTrain <- rep(NA, nrow(DataList_train$survival))
      predVal <- rep(NA, nrow(DataList_val$survival))
    }else{
      mole.vars <- rownames(m2efm$initial_model$glmnet.fit$beta)
      predTrain <- predict(m2efm, comb, clin, colnames(clin))
      predTrain <- exp(predTrain[, 1]) %>% as.numeric()

      methVal <- as.data.frame(t(DataList_val$meth450))
      methVal <- beta2m(methVal[probes,])

      moleVal <- as.data.frame(cbind(DataList_val$mRNATPM_map, t(methVal)))[, mole.vars]
      clinVal <- DataList_val$clinical
      colnames(clinVal) <- colnames(clin)
      predVal <- predict(m2efm, moleVal, clinVal, colnames(clinVal))
      predVal <- exp(predVal[, 1]) %>% as.numeric()
    }
    predTrain <- as.data.frame(cbind(predTrain, as.matrix(DataList_train$survival))) %>% `rownames<-`(rownames(DataList_train$survival))
    predVal <- as.data.frame(cbind(predVal, as.matrix(DataList_val$survival))) %>% `rownames<-`(rownames(DataList_val$survival))

    return(list(Train = predTrain, Val = predVal))
  }

  ### run the method
  lapply(allFiles, function(file){
    print(file)

    if(!dir.exists(file.path(resPath, "M2EFM", file))){
      dir.create(file.path(resPath, "M2EFM", file))
    }
    if(!dir.exists(file.path(timerecPath, "M2EFM", file))){
      dir.create(file.path(timerecPath, "M2EFM", file))
    }

    DataList <- readRDS(file.path(datPath, paste0(file, ".rds")))
    dataTypes <- c("mRNATPM_map", "miRNA", "cnv", "meth450", "clinical", "survival")
    dataTypesUsed <- c("mRNATPM_map", "meth450", "clinical", "survival")
    DataList <- DataList[dataTypes]
    commonSamples <- Reduce(intersect, lapply(DataList, rownames)) %>% unique()
    DataList <- lapply(dataTypesUsed, function(dataType){
      dat <- DataList[[dataType]][commonSamples, ]
      if (!dataType %in% c("clinical", "survival")){
        if (dataType == 'mRNATPM_map'){
          dat[is.na(dat)] <- 0
        }else{
          dat <- dat[, setdiff(colnames(dat), Probes_remove)]
          dat <- impute_meth(dat)
        }
        if (max(dat) > 100) {
          dat <- log2(dat + 1)
        }
      }
      dat
    }) %>% `names<-` (dataTypesUsed)

    survival <- DataList$survival
    ### 5 times of 10 fold cross-validation
    lapply(c(1:5), function(time){
      print(paste0('Running Time: ', time))
      if (!dir.exists(file.path(resPath, "M2EFM", file, paste0('Time', time)))) {
        dir.create(file.path(resPath, "M2EFM", file, paste0('Time', time)))
      }
      if (!dir.exists(file.path(timerecPath, "M2EFM", file, paste0('Time', time)))) {
        dir.create(file.path(timerecPath, "M2EFM", file, paste0('Time', time)))
      }
      set.seed(time)
      all_folds <- vfold_cv(survival, v = 10, repeats = 1, strata = status)
      all_folds <- lapply(1:10, function(fold) {
        patientIDs <- rownames(survival)[all_folds$splits[[fold]]$in_id]
      })

      lapply(1:10, function(fold) {
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

        write.csv(Res$Train, file.path(resPath, "M2EFM", file, paste0('Time', time), paste0("Train_Res_", fold, ".csv")), row.names = T)
        write.csv(Res$Val, file.path(resPath, "M2EFM", file, paste0('Time', time), paste0("Val_Res_", fold, ".csv")), row.names = T)
        write.csv(time_df, file.path(timerecPath, "M2EFM", file, paste0('Time', time), paste0("TimeRec_", fold, ".csv")), row.names = T)
      })
    })
    # write.csv(alltimes, file.path(timerecPath, "M2EFM", file, "TimeRec.csv"), row.names=T)
    return(NULL)
  })
}

