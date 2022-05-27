# -*- coding: utf-8 -*-

### 강의 주제: 스태킹 모델의 구축

## 스태킹
## - 다수개의 머신러닝 모델이 예측한 값을 학습하여 결과를 반환하는 방법
##
## 스태킹 모델 구축
## - 앙상블: 다수개의 머신러닝 모델의 예측값을 취합하여 평균 또는
##         다수결의 원칙을 이용하여 예측하는 모델
## - 앙상블 사용 이유: 일반화의 성능을 극대화하기 위해 
##                 (예측 성능의 분산을 감소시킬 수 있음)

from pyexpat import model
import pandas as pd
pd.options.display.max_columns = 100
import numpy as np

## 1. 데이터 적재
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

## 2. 데이터 생성 
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(X.head())
print(X.info())
print(X.describe())

## 3. 데이터 전처리 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 4. 앙상블 구현
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

## 모델 객체 생성 및 학습 
model_lr = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
model_kn = KNeighborsClassifier().fit(X_train_scaled, y_train)
model_dt = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, y_train)

score = model_lr.score(X_train_scaled, y_train)
print(f'학습(lr): {score}')
score = model_kn.score(X_train_scaled, y_train)
print(f'학습(kn): {score}')
score = model_dt.score(X_train_scaled, y_train)
print(f'학습(dt): {score}')

score = model_lr.score(X_test_scaled, y_test)
print(f'테스트(lr): {score}')
score = model_kn.score(X_test_scaled, y_test)
print(f'테스트(kn): {score}')
score = model_dt.score(X_test_scaled, y_test)
print(f'테스트(dt): {score}')

##
## 5. 스태킹 구현 
##

## 각 모델의 예측 결과를 취합
pred_lr = model_lr.predict(X_train_scaled)
pred_kn = model_kn.predict(X_train_scaled)
pred_dt = model_dt.predict(X_train_scaled)
pred_stack = np.array([pred_lr, pred_kn, pred_dt])

## pred_stack과 y_train의 shape이 맞지 않음
## => 전치 행렬을 통해 해결 
pred_stack = pred_stack.T

## 앙상블 모델 객체 생성 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,
                               max_depth=None,
                               max_samples=0.5,
                               max_features=0.3,
                               random_state=1)

## 앙상블 모델을 학습
## - 각 모델을 취합한 결과를 이용하여 학습 
model.fit(pred_stack, y_train)

## 앙상블 모델 평가 
score = model.score(pred_stack, y_train)
print(f'학습: {score}')

## 각 모델 평가 
## - 테스트 데이터에 대한 각 모델들의 예측 값을 취합 
pred_lr = model_lr.predict(X_test_scaled)
pred_kn = model_kn.predict(X_test_scaled)
pred_dt = model_dt.predict(X_test_scaled)
pred_stack = np.array([pred_lr, pred_kn, pred_dt]).T

score = model.score(pred_stack, y_test)
print(f'테스트: {score}')
