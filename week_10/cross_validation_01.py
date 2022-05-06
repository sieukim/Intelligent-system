# -*- coding: utf-8 -*-

##### 강의 주제: 머신러닝을 사용하여 데이터를 분석하는 과정

### 1. 데이터 적재
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

### 2. 데이터 분할
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=0.3,
                                               stratify=y,
                                               random_state=1)

### 3. 데이터 전처리
from sklearn.preprocessing import StandardScaler

### 3-1. 스케일러 객체 생성
scaler = StandardScaler()

### 3-2. 스케일러 학습
scaler.fit(X_train)

### 3-3. 스케일 처리된 학습 데이터
X_train_scaled = scaler.transform(X_train)

### 3-4. 스케일 처리된 테스트 데이터
X_test_scaled = scaler.transform(X_test)

### 4. 머신러닝 모델 구축
from sklearn.linear_model import LogisticRegression

### 4-1. 머신러닝 모델 객체 생성
model = LogisticRegression(C=1.0,
                           n_jobs=-1,
                           random_state=1)

### 5. 머신러닝 모델 학습
model.fit(X_train_scaled, y_train)

### 6. 머신러닝 모델 평가
score = model.score(X_train_scaled, y_train)
print(f'(MODEL) TRAIN SCORE : {score}')

score = model.score(X_test_scaled, y_test)
print(f'(MODEL) TEST SCORE : {score}')










