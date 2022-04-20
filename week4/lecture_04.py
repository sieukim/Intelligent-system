# -*- coding: utf-8 -*-

import pandas as pd

### 강의 주제: 총 정리

### 1. 데이터 적재
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

### 2. 데이터 설정
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

### 3. 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, 
                                                    random_state=1)

### 4. 차원 확장
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)

X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)

### 5. 스케일 처리
### - 정규화: 데이터를 구성하는 각 컬럼의 평균을 0, 표준편차는 1로 조정
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### 6. 머신러닝 모델 구축
from sklearn.linear_model import LinearRegression, Lasso, Ridge

## 모델 객체 생성
model = LinearRegression(n_jobs=-1)
lasso_model = Lasso(alpha=0.01, max_iter=1000000, random_state=1)
ridge_model = Ridge(alpha=1., random_state=1)

## 모델 학습
model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

## 모델 평가
score = model.score(X_train, y_train)
print(f'Train(LR): {score}')

score = lasso_model.score(X_train, y_train)
print(f'Train(Lasso): {score}')

score = ridge_model.score(X_train, y_train)
print(f'Train(Ridge): {score}')

print('----------------------')

score = model.score(X_test, y_test)
print(f'Test(LR): {score}')

score = lasso_model.score(X_test, y_test)
print(f'Test(Lasso): {score}')

score = ridge_model.score(X_test, y_test)
print(f'Test(Ridge): {score}')
