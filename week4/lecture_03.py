# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

### 강의 주제: 선형 모델의 성능 향상

### 선형 모델의 성능 향상 방법
### 1. 스케일 전처리(정규화/일반화)
### - 선형 모델은 각 컬럼에 가중치를 할당하는 방식인데,
###   각 컬럼의 스케일이 차이가 나면 가중치 적용에 어려움
###   따라서, 스케일 전처리가 필요
###
### 2. 차원 확장
### - 선형 모델은 기본적으로 1차원 직선을 사용하여 데이터를 예측
### - 차원을 확장함으로써 성능을 향상시킴


###
### 1. 데이터 설정
###

### 1-1. 설명변수 설정
X = np.arange(1, 11).reshape(-1, 1)

### 1-2. 종속변수 설정
y = [5, 8, 10, 9, 7, 5, 3, 6, 9, 10]

### 1-3. 데이터 분포 확인
plt.plot(X, y, 'xb')


###
### 2. 머신러닝 모델 구축
###

from sklearn.linear_model import LinearRegression

### 2-1. 머신러닝 모델 객체 생성
model = LinearRegression()

### 2-2. 머신러닝 모델 객체 학습
model.fit(X, y)

### 2-3. 머신러닝 모델 객체 평가
print(model.score(X, y))

### 2-4. 머신러닝 모델 객체 예측
pred = model.predict(X)

### 2-5. 예측 확인
plt.plot(X, pred, '-g')


###
### 3. 차원 확장
### 

### PolynomialFeatures: 다항식 생성 클래스
### PolynomialFeatures(degree=생성할 다항식의 차수,
###                    include_bias=상수항(bias) 생성 여부)
from sklearn.preprocessing import PolynomialFeatures

### 3-1. 기본 다항식 형태 생성
poly = PolynomialFeatures(degree=2, include_bias=False)

### 3-2. 1차원의 데이터를 2차원으로 확장
X_poly = poly.fit_transform(X)

print(X_poly)

### 3-3. 머신러닝 모델 객체 학습
model.fit(X_poly, y)

### 3-4. 머신러닝 모델 객체 예측
pred = model.predict(X_poly)

### 3-5. 예측 확인
plt.plot(X, pred, '-g')
