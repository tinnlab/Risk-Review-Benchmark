library(tidyverse)
library(parallel)

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)
Sys.setenv(OMP_NUM_THREADS = 1,
           OPENBLAS_NUM_THREADS = 1,
           MKL_NUM_THREADS = 1,
           VECLIB_MAXIMUM_THREADS = 1,
           NUMEXPR_NUM_THREADS = 1)

resPath <- "/nfs/blanche/share/daotran/SurvivalPrediction/Results/Priority-Lasso"
allFolds <- list.files(resPath)

allResTrain <- mclapply(allFolds, mc.cores = 10, function(fold){
  print(fold)
  allFiles <- list.files(file.path(resPath, fold))
  allFiles <- allFiles[grep("Train", allFiles)]
  Res <- lapply(allFiles, function(file){
    read.table(file.path(resPath, fold, file), sep = ",", header = 1)
  }) %>% `names<-`(allFiles)
  Res
}) %>% `names<-` (allFolds)

