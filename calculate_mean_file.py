# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:54:36 2024

@author: zhengke
"""
import rpy2
from rpy2.robjects import r
from rpy2.robjects.packages import importr 
import calculate_propensity_score as cps
import numpy as np
import pandas as pd
import os
import csv
importr('dplyr')
r_code="""
options (warn = -1)
# CTS department data, the unit of ICU stay is half day (12h)
df_cts <- read.csv("inspire_CTS_0_120_icuhalfday.csv")
# X: features
X <- df_cts[,4:40]
# W: treatments as factor
W <- factor(df_cts[,41])
# Y: outcomes
Y <- df_cts[,1]
# C: translation, to construct maximize problem, where all Y are positive
C <- max(Y)+1
# outcomes after translation
df_cts[,1] <- C-Y
set.seed(123)
# Initialize a null list
# index_list: list of J sub lists, each sub list contains row indexs of samples with real treatment j
index_list <- list()
# obtain unique value of delay_group 
unique_values <- unique(df_cts$delay_group)
# create index_list
for (value in unique_values) {
  indices <- which(df_cts$delay_group == value)
  index_list[[as.character(value)]] <- indices
}
# index_j is a list contains row indexs of samples with real treatment j
# sample(): randomize each index_j list to be divided in k folds
index1<-sample(index_list$`1`)
index2<-sample(index_list$`2`)
index3<-sample(index_list$`3`)
index4<-sample(index_list$`4`)
# indexj_fold: list of 10 sub lists, each sub list k contains row indexs of samples with real treatment j apart from samples in fold k
index1_fold <- list()
# fold_index: vector of length N (sample size), storing the fold index k that each sample belongs to
fold_index<-rep(0, 858)
# 10-fold, almost average division
for (i in 1:10) {
  if (i<10){
  fold <- index1[-((59*(i-1)+1):(59*i))]
  index1_fold[[paste0("fold", i)]] <- fold
  for (j in (59*(i-1)+1):(59*i)){
    fold_index[index1[j]]<-i
  }
  }
  else{
  fold <- index1[-((59*(i-1)+1):(59*i+1))]
  index1_fold[[paste0("fold", i)]] <- fold
  for (j in (59*(i-1)+1):(59*i+1)){
    fold_index[index1[j]]<-i
  }
  }
}
index2_fold <- list()
for (i in 1:10) {
  if (i<5){
  fold <- index2[-((14*(i-1)+1):(14*i))]
  index2_fold[[paste0("fold", i)]] <- fold
  for (j in (14*(i-1)+1):(14*i)){
    fold_index[index2[j]]<-i
  }}
  else{
  fold <- index2[-((15*(i-1)-3):(15*i-4))]
  index2_fold[[paste0("fold", i)]] <- fold
  for (j in (15*(i-1)-3):(15*i-4)){
    fold_index[index2[j]]<-i
  }
  }
}
index3_fold <- list()
for (i in 1:10) {
  if (i<9){
  fold <- index3[-((7*(i-1)+1):(7*i))]
  index3_fold[[paste0("fold", i)]] <- fold
  for (j in (7*(i-1)+1):(7*i)){
    fold_index[index3[j]]<-i
  }
  }
  else{
  fold <- index3[-((7*(i-1)+1):(7*i+1))]
  index3_fold[[paste0("fold", i)]] <- fold
  for (j in (7*(i-1)+1):(7*i+1)){
    fold_index[index3[j]]<-i
  }
  }
}
index4_fold <- list()
for (i in 1:10) {
  fold <- index4[-((5*(i-1)+1):(5*i))]
  index4_fold[[paste0("fold", i)]] <- fold
  for (j in (5*(i-1)+1):(5*i)){
    fold_index[index4[j]]<-i
  }
}
# for each fold, combine list of each treatment j
index_fold <- list()
for (i in 1:10) {
  index_fold[[paste0("fold", i)]] <- c(index1_fold[i],index2_fold[i],index3_fold[i],index4_fold[i])
}
# model_list: list of prediction models
model_list <- list()
# train 10个模型
for (i in 1:10){
  index_cts<-df_cts[unlist(index_fold[i]),]
  index_data<-data.frame(
  y1=index_cts$icu_stay,
  x1=factor(index_cts$delay_group),
  x2=index_cts$age,
  x3=index_cts$height,
  x4=index_cts$weight,
  x5=index_cts$asa,
  x6=index_cts$nibp_sbp,
  x7=index_cts$nibp_dbp,
  x8=index_cts$spo2,
  x9=index_cts$hr,
  x10=index_cts$albumin,
  x11=index_cts$alp,
  x12=index_cts$alt,
  x13=index_cts$creatinine,
  x14=index_cts$glucose,
  x15=index_cts$hct,
  x16=index_cts$aptt,
  x17=index_cts$potassium,
  x18=index_cts$sodium,
  x19=index_cts$bun,
  x20=index_cts$wbc,
  x21=index_cts$sex,
  x22=index_cts$hypertension,
  x23=index_cts$coronary_artery_disease,
  x24=index_cts$prior_myocardial_infarction,
  x25=index_cts$congestive_heart_failure,
  x26=index_cts$aortic_stenosis,
  x27=index_cts$atrial_fibrillation,
  x28=index_cts$prior_stroke,
  x29=index_cts$transient_ischemic_attack,
  x30=index_cts$peripheral_artery_disease,
  x31=index_cts$deep_venous_thrombosis,
  x32=index_cts$pulmonary_embolism,
  x33=index_cts$diabetes_mellitus,
  x34=index_cts$chronic_kidney_disease,
  # x35=index_cts$ongoing_dialysis,
  # x36=index_cts$pulmonary_hypertension,
  x37=index_cts$chronic_obstructive_pulmonary_disease,
  x38=index_cts$asthma,
  # x39=index_cts$obstructive_sleep_apnea,
  # x40=index_cts$cirrhosis,
  x41=index_cts$gastro_esophageal_reflux,
  x42=index_cts$anemia
  #x43=index_cts$dementia
  )
  index_model<-lm(y1~x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 
                  # + x35 + x36 
                  + x37 + x38 
                  # + x39 + x40 
                  + x41 + x42 
                  # + x43
                  ,data=index_data)
  model_list[[i]] <- index_model
}
# calculate_mean: function that returns predicted mean E[Y(j)|X=x]
calculate_mean<-function(s,j) {
  s_cts<-df_cts[s,]
  df_s<-data.frame(
  y1=s_cts$icu_stay,
  # x1=s_cts$delay_group,
  x1=factor(j),
  x2=s_cts$age,
  x3=s_cts$height,
  x4=s_cts$weight,
  x5=s_cts$asa,
  x6=s_cts$nibp_sbp,
  x7=s_cts$nibp_dbp,
  x8=s_cts$spo2,
  x9=s_cts$hr,
  x10=s_cts$albumin,
  x11=s_cts$alp,
  x12=s_cts$alt,
  x13=s_cts$creatinine,
  x14=s_cts$glucose,
  x15=s_cts$hct,
  x16=s_cts$aptt,
  x17=s_cts$potassium,
  x18=s_cts$sodium,
  x19=s_cts$bun,
  x20=s_cts$wbc,
  x21=s_cts$sex,
  x22=s_cts$hypertension,
  x23=s_cts$coronary_artery_disease,
  x24=s_cts$prior_myocardial_infarction,
  x25=s_cts$congestive_heart_failure,
  x26=s_cts$aortic_stenosis,
  x27=s_cts$atrial_fibrillation,
  x28=s_cts$prior_stroke,
  x29=s_cts$transient_ischemic_attack,
  x30=s_cts$peripheral_artery_disease,
  x31=s_cts$deep_venous_thrombosis,
  x32=s_cts$pulmonary_embolism,
  x33=s_cts$diabetes_mellitus,
  x34=s_cts$chronic_kidney_disease,
  # x35=s_cts$ongoing_dialysis,
  # x36=s_cts$pulmonary_hypertension,
  x37=s_cts$chronic_obstructive_pulmonary_disease,
  x38=s_cts$asthma,
  # x39=s_cts$obstructive_sleep_apnea,
  # x40=s_cts$cirrhosis,
  x41=s_cts$gastro_esophageal_reflux,
  x42=s_cts$anemia
  # x43=s_cts$dementia
  )
  return(predict(model_list[[fold_index[s]]], df_s))}
N=858
num_j=4
mean_matrix <- matrix(nrow = N, ncol = num_j)
# write csv
for (s in 1:N) {
  for (j in 1:num_j) {
    mean_value <- calculate_mean(s,j)
    mean_matrix[s,j] <- mean_value
  }
}
write.table(mean_matrix, file = "calculate_mean.csv", sep = ",", row.names = FALSE, col.names = FALSE)
    """
r(r_code)
calculate_mean = r['calculate_mean']
N=858
model_dict={}
df_CTS = pd.read_csv("inspire_CTS_0_120_icuhalfday.csv")
x = df_CTS.iloc[:, 3:40].values
for i in range(10):
    X_train=pd.DataFrame(r('df_cts[unlist(index_fold['+str(i+1)+']),4:40]')).T.values
    D_train=np.array(r('df_cts[unlist(index_fold['+str(i+1)+']),41]'))
    model_dict[i]=cps.train_propensity_model(X_train,D_train)
    
with open('propensity_score.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for s in range(N):
        row = []
        for j in range(4):
            propensity_score = cps.propensity_score(model_dict[r('fold_index[' + str(s+1) + ']')[0]-1], x[s], j)
            row.append(propensity_score[0])
        writer.writerow(row)