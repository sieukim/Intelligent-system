# -*- coding: utf-8 -*-

## 강의 주제: 비지도 학습 - 데이터 전처리

import pandas as pd
pd.options.display.max_columns = 100

## 1. 데이터 적재
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

## 2. 데이터 설정 
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(X.head())
print(X.info())
print(X.describe())

##
## 3. 군집 분석 수행 
##

## 머신러닝 모델 학습 결과, 가중치가 할당되지 않거나 굉장히 작은 칼럼들에 
## 대해 새로운 특성을 생성한다. 이때 해당 칼럼들에 대한 최적의 군집 개수를
## 찾고, 이를 이용하여 모델을 생성하고 예측하여 값을 대체한다. 

X_part = X[['radius error',     # 미리 학습하고 알아낸
           'compactness error', # 가중치가 굉장히 작은
           'concavity error']]  # 칼럼들 

## 최적의 군집 개수 검색
from sklearn.cluster import KMeans
values = []
for i in range(1, 15):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_part)
    values.append(km.inertia_) 
print(values)

## inertia_ 분포 확인 
import matplotlib.pyplot as plt
plt.plot(range(1, 15), values, marker='o')
plt.xlabel('numbers of cluster')
plt.ylabel('inertia_')
plt.show()

## => inertia_ 분포 확인 결과 최적의 군집 개수는 5개이다. 

## 최적의 군집 개수를 이용하여 KMeans 모델을 생성
km = KMeans(n_clusters=5,   
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)

## 해당 칼럼들에 대해 학습 수행 
km.fit(X_part) 

## 해당 칼럼들에 대한 예측을 이용하여 새로운 특성을 생성 
X['cluster_result'] = km.predict(X_part) 

## 기존 칼럼들 삭제 
del X['radius error']
del X['compactness error']
del X['concavity error']

print(X.info())

## 4. 데이터 전처리  

## 데이터 분할 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=1)

## 스케일 처리 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 5. 모델 학습 

## 모델 객체 생성 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, 
                           class_weight='balanced',
                           random_state=0)

## 모델 학습
model.fit(X_train_scaled, y_train)

## 모델 평가 
score = model.score(X_train_scaled, y_train)
print(f'학습: {score}')
score = model.score(X_test_scaled, y_test)
print(f'테스트: {score}')

## 가중치 확인 
print(f'학습된 가중치: \n{model.coef_}')

 