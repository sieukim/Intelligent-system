# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 부스팅 AdaBoosting

### AdaBoosting: 데이터 중심의 부스팅 방법론. 직전 모델 모델이 예측한 데이터에
###              가중치를 부여하는 방법


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
### 4. 머신러닝 모델 구축 with AdaBoostingClassfier
###

### 4-1. 앙상블 클래스 로딩
from sklearn.ensemble import AdaBoostClassifier

### 4-2. 내부에서 사용할 머신러닝 알고리즘 클래스 로딩
from sklearn.linear_model import LogisticRegression

base_model = LogisticRegression(C=0.001,
                                class_weight='balanced',
                                n_jobs=-1,
                                random_state=1)

### 4-3. base_estimator 정의 (사용할 머신러닝 모델 정의)
base_estimator = base_model


### 4-4. 머신러닝 모델 객체 생성
model = AdaBoostClassifier(base_estimator=base_estimator,
                           n_estimators=100,
                           learning_rate=1.0,
                           random_state=1)

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