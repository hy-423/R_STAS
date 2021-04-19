library(readxl)
library(tidyverse)
library(glmnet)
library(caret)
library(xgboost)
library(Matrix)
library(data.table)
library(randomForest)
library(pROC)
library(doParallel)
library(survival)
library(psych)
library(xgboost)

library(doParallel)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

lc <- read_excel("R_result20200902.xlsx", sheet = "label20200902")
lc <- as.data.frame(lc)
# lc <- read.csv("label_2_20200713.csv")

inclusion <- read_excel("R_result20200902.xlsx", sheet = "inclusion")
lc <- lc[inclusion$include == 1,]

colnames(lc)[1:10]
lc <- lc[!is.na(lc[,10]),]
dim(lc)

Pstas <- NULL
for (k in 3:ncol(lc)){
  # Pstas <- c(Pstas, roc(lc$stas ~ lc[,k])$auc)
  Pstas <- c(Pstas, wilcox.test(lc[,k] ~ lc$stas)$p.value)
}
min(Pstas)
sum(Pstas < 0.05)

first <- 2
last <- ncol(lc)

Repeat = 5
FoldNum = 5
subsets <- c(5:100)

Param <- list(objective = "binary:logistic",
              eval_metric = "auc")
cv.nround = 20
Treenum = 20

#### rfe xg stas ####

ThAUC.stas <- NULL
ThAUC.stas.values <- NULL
for (i in rep(0.05,10)){

  lc.IF <- lc[,3:ncol(lc)]
  lc.stas <- cbind.data.frame(lc[,2], lc.IF[,Pstas < i])
  colnames(lc.stas)[1] <- "stas"


  rfe.xg.stasAll <- NULL
  for (j in 1:Repeat){
    set.seed(NULL)
    rfe.xg.stas <- NULL
    Fold <- createFolds(factor(lc.stas[,1]), k=FoldNum)
    for (k in 1:FoldNum){
      print(paste(i,j,k))

      ctrl <- rfeControl(functions = rfFuncs,
                         method = "repeatedcv",
                         number = FoldNum,
                         repeats = Repeat,
                         verbose = FALSE)

      lmProfile <- rfe(lc.stas[-Fold[[k]],first:ncol(lc.stas)], factor(lc.stas$stas[-Fold[[k]]]),
                       sizes = subsets,
                       rfeControl = ctrl)

      Pred <- NULL
      for (Pr in 1:80){
        Pre <- str_which(colnames(lc.stas), predictors(lmProfile)[Pr])
        Pred <- c(Pre, Pred)
      }
      if(length(predictors(lmProfile)) > 1){
        lc2 <- lc.stas[,sort(Pred)]

        data=as.matrix(lc2[-Fold[[k]],])
        dtrain <-xgb.DMatrix(data=data, label = as.integer(lc.stas$stas[-Fold[[k]]]))
        bst <- xgb.train(param = Param, data = dtrain,  nrounds = Treenum)
        dtest <-xgb.DMatrix(data=as.matrix(lc2[Fold[[k]],]), label = as.integer(lc.stas$stas[Fold[[k]]]))
        pred0 <- predict(bst, dtest)
        rfe.xg.stas <- c(rfe.xg.stas, pred0)}
      else{
        lc2 <- lc.stas[,sort(Pred)]

        data=as.matrix(lc2[-Fold[[k]]])
        dtrain <-xgb.DMatrix(data=data, label = as.integer(lc.stas$stas[-Fold[[k]]]))
        bst <- xgb.train(param = Param, data = dtrain,  nrounds = Treenum)
        dtest <-xgb.DMatrix(data=as.matrix(lc2[Fold[[k]]]), label = as.integer(lc.stas$stas[Fold[[k]]]))
        pred0 <- predict(bst, dtest)
        rfe.xg.stas <- c(rfe.xg.stas, pred0)
      }
    }
    rfe.xg.stasAll <- rbind(rfe.xg.stasAll, rfe.xg.stas[order(unlist(Fold))])
  }
  print(roc(lc.stas$stas ~ apply(rfe.xg.stasAll, 2, mean))$auc)
  ThAUC.stas <- c(ThAUC.stas, roc(lc.stas$stas ~ apply(rfe.xg.stasAll, 2, mean))$auc)
  ThAUC.stas.values <- rbind(ThAUC.stas.values, apply(rfe.xg.stasAll, 2, mean))
}

ThAUC.stas

write.csv(ThAUC.stas, "AUCs.csv")
write.csv(ThAUC.stas.values, "AUCs_values.csv")
