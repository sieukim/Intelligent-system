# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 앙상블 RandomForestClassifier

### RandomForest:  Bagging에 Decision Tree를 조합하여 사용하는 방법
### 주로 사용하는 구조이기때문에 하나의 앙상블 모형으로 구현해놓은 클래스

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
### 4. 머신러닝 모델 구축 with RandomForestClassifier
###

### 4-1. 앙상블 클래스 로딩
from sklearn.ensemble import RandomForestClassifier

### 4-2. 머신러닝 모델 객체 생성
model = RandomForestClassifier(n_estimators=50,
                               max_depth=None,
                               max_samples=0.3,
                               max_features='auto',
                               n_jobs=-1,
                               random_state=1)

### 4-3. 머신러닝 모델 객체 학습
model.fit(X_train, y_train)

### 4-4. 머신러닝 모델 객체 평가
sc = model.score(X_train, y_train)
print(f'Train: {sc}')

sc = model.score(X_test, y_test)
print(f'Test: {sc}')

### 4-5. 머신러닝 모델 객체 예측
pred = model.predict(X_test[:5])
print(f'Pred: {pred}')