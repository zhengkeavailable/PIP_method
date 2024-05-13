# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:12:58 2024

@author: zhengke
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

## 计算\hat{e}_j(X)
# 训练multinomial model
def train_propensity_model(X_train, D_train):
    label_encoder = LabelEncoder()
    D_train_encoded = label_encoder.fit_transform(D_train)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, D_train_encoded)
    return model

# 计算propensity score
def propensity_score(model, X, j):
    # 计算每个类别的概率
    X=X.reshape(1,-1)
    probabilities = model.predict_proba(X)
    ps = probabilities[:, j]
    return ps