# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 서포트 벡터 머신 분류 평가

### 1. 데이터 적재
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()


### 2. 데이터 설정
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


### 3. 데이터 확인

### 3-1. 설명변수 EDA
print(X.info())

### 3-2. 설명변수 결측 데이터 정보
print(X.isnull().sum())

### 3-3. 설명변수 통계 정보
print(X.describe(include="all"))

### 3-4. 종속변수 
print(y.head())


### 4. 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1,
                                                    stratify=y)


### 5. 머신러닝 모델 구축
from sklearn.svm import SVC

### 5-1. 머신러닝 모델 객체 생성
model = SVC(C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=1)

### 5-2. 머신러닝 모델 객체 학습
model.fit(X_train, y_train)

### 5-3. 머신러닝 모델 객체 평가
sc = model.score(X_train, y_train)
print(f'Train: {sc}')

sc = model.score(X_test, y_test)
print(f'Test: {sc}')

### 5-4. 머신러닝 모델 객체 예측
pred = model.predict(X_train[:5])
print(pred)

###
### 6. 분류 모델의 평가 방법
###

### 6-1. 혼동 행렬

### confusion_matrix: 혼동행렬로, 실제 데이터와 예측 데이터
### 간의 관계를 나타낸 행렬
from sklearn.metrics import confusion_matrix

pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)

### [혼동 행렬]
###         예측 P  |  예측 N
### 실제 P    TP        FN
### 실제 N    FP        TN
###
### TP(True Positive)
### FP(False Positive) 
### FN(False Negative)
### TN(True Negative)
print(cm) 

### [혼동 행렬 결과]
###         예측 0  |  예측 1
### 실제 0    126        22
### 실제 1     17       233
###
###
### 정확도
### - 정답을 예측한 데이터 / 전체 데이터
### - (TP + TN) / Total
### - 분류 클래스의 비율이 동일한 경우에만 사용
###
### 정밀도
### - 정답을 예측한 데이터 / 각 클래스 데이터
### - TP / (TP + FP)
###
### 재현율
### - 정답을 예측한 데이터 / 학습 데이터셋
### - TP / (TP + FN)

### 6-2. 정확도
from sklearn.metrics import accuracy_score

### - 정확도 = (126 + 233) / (126 + 22 + 17 + 233)
acs = accuracy_score(y_train, pred)
print(f'acs: {acs}\n(tp + tn) / total = {(126 + 233) / (126 + 22 + 17 + 233)}')

### 6-3. 정밀도
from sklearn.metrics import precision_score

### - 0에 대한 정밀도 = 126 / (126 + 17)
ps = precision_score(y_train, pred, pos_label=0)
print(f'ps(0) = {ps}\ntp / (tp + fp) = {126 / (126 + 17)}')

### - 1에 대한 정밀도 = 233 / (233 + 22)
ps = precision_score(y_train, pred, pos_label=1)
print(f'ps(1) = {ps}\ntp / (tp + fp) = {233 / (233 + 22)}')

### 6-4. 재현율
from sklearn.metrics import recall_score

### - 0에 대한 재현율 = 126 / (126 + 22)
rs = recall_score(y_train, pred, pos_label=0)
print(f'rs(0) = {rs}\ntp / (tp + fn) = {126 / (126 + 22)}')

### - 1에 대한 재현율 = 233 / (17 + 233)
rs = recall_score(y_train, pred, pos_label=1)
print(f'rs(0) = {rs}\ntp / (tp + fn) = {233 / (17 + 233)}')