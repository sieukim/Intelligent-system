# -*- coding: utf-8 -*-
"""
Created on Fri Apr 8 00:04:53 2022

@author: potato
"""

import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 앙상블 VotingClassfier

### 앙상블(Emsemble): 여러개의 머신러닝 알고리즘을 결합하여 각 모델이 예측한
### 결과를 취합/부스팅하는 방법을 통해 예측을 수행하는 방법(론)

### 앙상블 구현 방식
###
### 1. 취합
### - 앙상블을 구성하는 내부의 각 모델이 서로 독립적으로 동작
### - 각 모델이 예측한 결과에 대해 다수결 방식 수행(분류)
### - 각 모델이 예측한 결과에 대해 평균값을 취함(회귀 분석)
### - 각 모델은 서로 연관성이 없음
### - 각 모델은 적절한 수준의 과적합을 수행할 필요가 있음
### - 학습과 예측의 수행 속도가 빠름 (각 모델이 독립적이라 병렬 처리 가능)
### - ex) Voting, Bagging, RandomForest
###
### 2. 부스팅
### - 앙상블을 구성하는 내부의 각 모델이 선형으로 연결되어 동작
### - 각 모델은 다음 모델에 영향을 줌
### - 각 모델에 강한 제약을 설정하여 점진적인 성능향상을 도모
### - 학습과 예측의 수행 속도가 느림 (각 모델이 선형으로 연결되어 앞의 모델이 
###   종료되어야 이후 모델이 시작하기때문) 
### - ex) AdaBoosting, GradientBoosting, XCBoosting, LightGBM

### 1. 데이터 적재
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()


### 2. 데이터 설정
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


### 3. 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

###
### 4. 머신러닝 모델 구축 with VotingClassfier
###

### 4-1. 앙상블 클래스 로딩
from sklearn.ensemble import VotingClassifier
 
### 4-2. 내부에서 사용할 머신러닝 알고리즘 클래스 로딩
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model1 = KNeighborsClassifier(n_jobs=-1)
model2 = LogisticRegression(n_jobs=-1, random_state=1)
model3 = DecisionTreeClassifier(random_state=1)

### 4-3. estimators 정의 (사용할 머신러닝 모델 정의)
estimators = [('knn', model1), ('lr', model2), ('dt', model3)]

### 4-4. 머신러닝 모델 객체 생성
model = VotingClassifier(estimators=estimators,
                         voting='hard',
                         n_jobs=-1)

### 4-5. 머신러닝 모델 객체 학습
model.fit(X_train, y_train)

### 4-6. 머신러닝 모델 객체 평가
sc = model.score(X_train, y_train)
print(f'Train: {sc}')

sc = model.score(X_test, y_test)
print(f'Test: {sc}')

### 4-7. 머신러닝 모델 객체 예측
pred = model.predict(X_test[:5])
print(f'Pred: {pred}')

### 4-8. 앙상블 내부에 존재하는 모든 모델에 대한 예측 
pred = model.estimators_[0].predict(X_test[:5])
print(f'Pred(KNN): {pred}')

pred = model.estimators_[1].predict(X_test[:5])
print(f'Pred(LR): {pred}')

pred = model.estimators_[2].predict(X_test[:5])
print(f'Pred(DT): {pred}')
