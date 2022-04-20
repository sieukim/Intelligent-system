# -*- coding: utf-8 -*-

import numpy as np

### 1. 1차원 배열 생성
### - numpy.arange([start, ] stop, [step, ] dtype=None)
### => start부터 시작하여 stop까지 step간격의 데이터를 이용하여 1차원 배열 생성
X = np.arange(1, 11)
print(X)

### 2. 1차원 배열을 2차원으로 변환
### - numpy.reshape(a, newshape, order='C')
### => a를 newshape차원으로 변환
### - newshape는 n, m으로 n행 m열을 의미함
### - n이 -1인 경우 => 원래 데이터의 행과 열의 개수 곱을 m으로 나눈 값을 n으로 결정
### - m이 -1인 경우 => 원래 데이터의 행과 열의 개수 곱을 n으로 나눈 값을 m으로 결정
X = X.reshape(-1, 1)
print(X)

### 3. 종속변수 생성
y = np.arange(10, 101, 10)
print(y)

### 4. 최근접 이웃 알고리즘 수행(회귀)
### - KNeighborsRegressor: 근접한 이웃의 클래스를 살피고 다수결에 따라 해당 데이터의
###   클래스를 결정하는 KNeighborsClassfier와 달리, KNeighborsRegreesor는 다수결이 
###   아닌 평균 값으로 데이터의 값을 결정
from sklearn.neighbors import KNeighborsRegressor

### 4-1. 최근접 이웃 알고리즘 객체 생성
model = KNeighborsRegressor(n_neighbors=2)

### 4-2. 최근접 이웃 알고리즘 학습 수행
model.fit(X, y)

### 4-3. 예측 수행 (학습 데이터 범위 내 값)
X_pred = [[3.7]]
pred = model.predict(X_pred)
print(pred)

### 4-4. 예측 수행(학습 데이터 범위 외 값)
X_pred = [[57.7]]
pred = model.predict(X_pred)

### !!최근접 이웃 알고리즘의 단점!!
### - 학습에 사용되어 저장된 값의 범위 내에서만 예측이 가능함 
print(pred) # 577이 예상됨. 하지만 95라는 값을 예측함

### 5. 선형 회귀 수행
from sklearn.linear_model import LinearRegression

### 5-1. 선형 회귀 객체 생성
model = LinearRegression()

### 5-2. 선형 회귀 학습 수행
model.fit(X, y)

### 5-3. 예측 수행 (학습 데이터 범위 내 값)
X_pred = [[3.7]]
pred = model.predict(X_pred)
print(pred)

### 5-4. 예측 수행(학습 데이터 범위 외 값)
X_pred = [[57.7]]
pred = model.predict(X_pred)
print(pred)
