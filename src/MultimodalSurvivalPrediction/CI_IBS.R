source("/data/daotran/Cancer_RP/Benchmark/Metrics/CI_Calib.R")

Path <- "/data/daotran/Cancer_RP/Benchmark/Output"
method <- "MultimodalSurvivalPrediction"
dataTypes <- c("mRNATPM")
folderSave <- paste(c(dataTypes), collapse = "_")
tissue <- "Lung"
datasets <- list.files(file.path(Path, method, folderSave, tissue))

allRes <- lapply(datasets, function(dataset){
  resPath <- file.path(Path, method, folderSave, tissue, dataset)
  truePath <- file.path(Path, method, folderSave, tissue, dataset, "TrueLabel.txt")

  RiskScore <- read.table(file.path(resPath, "RiskScore.txt"))
  # SurvProb <- read.table(file.path(resPath, "SurvivalProb.txt"), header = 1)
  trueSurv <- read.table(truePath, header = 1)

  # trueSurv <- trueDat[rownames(RiskScore), ]
  truelabel <- Surv(trueSurv$time, trueSurv$status)

  CIndex <- coxph_cal(as.matrix(RiskScore), truelabel)$cIndex
  pValue <- coxph_cal(as.matrix(RiskScore), truelabel)$pValue
  # ICI_5y <- c()
  # for (i in 1:5){
  #   ICI_temp <- ModCal_Measure(as.matrix(SurvProb[, i]), trueSurv, 12*i)["ICI"]
  #   ICI_5y <- c(ICI_5y, ICI_temp)
  # }
  # ICI_5y <- as.matrix(ICI_5y)
  # ICI_5y <- rbind(ICI_5y, mean(ICI_5y[, 1]))
  # list(CIndex = CIndex, pValue = pValue, ICI = ICI_5y)

  list(CIndex = CIndex, pValue = pValue)
}) %>% `names<-` (datasets)

allCI <- lapply(names(allRes), function(dataset){
  Res <- allRes[[dataset]]
  CI <- Res$CIndex
}) %>% do.call(what=c) %>% as.data.frame() %>% `rownames<-` (names(allRes))

allpV <- lapply(names(allRes), function(dataset){
  Res <- allRes[[dataset]]
  pv <- Res$pValue
}) %>% do.call(what=c) %>% as.data.frame() %>% `rownames<-` (names(allRes))

# allICI <- lapply(names(allRes), function(dataset){
#   Res <- allRes[[dataset]]
#   ICI <- Res$ICI
# }) %>% do.call(what=rbind) %>% as.data.frame()

