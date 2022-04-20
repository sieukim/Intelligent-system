# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:10:50 2022

@author: potato
"""

import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 회귀 분석

### 회귀 분석
### - 머신러닝이 연속된 수치형의 데이터를 예측할 때 사용하는 기법


###
### 1. 데이터 적재(로딩)
###

### 1-1. 캘리포니아 집값 데이터셋
from sklearn.datasets import fetch_california_housing

### 1-2. 데이터 적재
data = fetch_california_housing()


###
### 2. 데이터 관찰(탐색)
###

### 2-1. 설명변수 설정
X = pd.DataFrame(data=data.data, columns=data.feature_names)

### 2-2. 종속변수 설정
y = pd.DataFrame(data=data.target)

### 2-3. 설명변수 분석
print(X.info())

### 2-4. 설명변수 내 결측 데이터 확인
print(X.isnull())
print(X.isnull().sum())


### 2-5. 설명변수의 모든 정보 확인
print(X.describe(include='all'))

### - 설명변수를 구성하는 각 특성들의 스케일을 반드시 확인
### - 스케일을 동일한 범위로 전처리 필요 
### - 각 특성들에 대해 산포도, 비율 등을 시각화하여 확인하거나 전처리해야 함

### 2-6. 종속변수 확인
print(y.head())


###
### 3. 데이터 분할
###

### - 회귀 분석을 위한 데이터셋의 경우 종속변수 내 값의 분포 비율을 유지할 필요가 없음
### - 비율이 중요한 경우에는 stratify를 사용하여 층화추출하면 됨

from sklearn.model_selection import train_test_split

### 3-1. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1)

### 3-2. 분할된 데이터 확인
print(X_train.shape, X_test.shape)
print(len(X_train), len(X_test), len(y_train), len(y_test))


###
### 4. 머신러닝 모델 구축
###

### LinearRegression: 선형 방정식을 기반으로 회귀 예측을 수행. 이 클래스의 학습은
### 설명변수를 구성하는 각 컬럼의 최적화된 가중치와 절편의 값을 계정하는 과정을 수행.
### y = x1 * w1 + x2 * w2 + ... + xn * wn + b
from sklearn.linear_model import LinearRegression

### 4-1. 머신러닝 모델 객체 생성
model = LinearRegression()

### 4-2. 머신러닝 모델 객체 학습
model.fit(X_train, y_train)

### 4-3. 머신러닝 모델 객체 평가
score = model.score(X_train, y_train)
print(f'Train: {score}')

score = model.score(X_test, y_test)
print(f'Test: {score}')

### 결정계수(R2) 계산 공식 = 1 - (정답과 예측값의 MSE / 정답과 정답의 평균값의 MSE)
### R2 == 0, 예측값이 전체 정답의 평균과 같은 경우 => 과소적합
### R2 == 1, 예측값과 정답이 완벽하게 일치
### R2 < 0, 예측값이 정답의 평균조차 예측하지 못 함 => 과소적합
### ==> 목표: 0.7 이상의 R2 값이 나오는 것

### 4-4. 머신러닝 모델 객체 예측

## 기울기(가중치)
coef = model.coef_
print(coef)

## 절편(바이어스)
intercept = model.intercept_

## 예측
pred = model.predict(X_train)
print(pred)

### 4-5. 머신러닝 모델 객체 평가
### - score 메소드 사용
### - 결정계수(R2): 데이터에 관계없이 동일한 결과를 사용하여 모델을 평가
### - 평균절대오차(MAE): |정답 - 예측값|의 평균(예측값의 신뢰 범위)
### - 평균절대오차비율: |정답 비율 - 예측값 비율|의 평균
### - 평균제곱오차(MSE): |정답 - 예측값| ** 2의 평균

## 결정계수
from sklearn.metrics import r2_score
## 평균절대오차
from sklearn.metrics import mean_absolute_error
## 평균절대오차비율
from sklearn.metrics import mean_absolute_percentage_error
## 평균제곱오차
from sklearn.metrics import mean_squared_error

## 평가를 위해 예측값이 필요
pred = model.predict(X_train)

## 결정계수
r2 = r2_score(y_train, pred)
print(f'R2: {r2}')
print(y_train.describe())

## 평균절대오차
mae = mean_absolute_error(y_train, pred)
print(f'MAE: {mae}')

## 평균절대오차비율
mape = mean_absolute_percentage_error(y_train, pred)
print(f'MAPE: {mape}')

## 평균제곱오차
mse = mean_squared_error(y_train, pred)
print(f'MSE: {mse}')


### 선형 모델이 학습한 가중치를 활용하여 중요도 파악하기
### - 특정 컬럼에 대한 가중치의 값이 0인 경우, 결과에 영향을 주지 않는 컬럼임
### - 특정 컬럼에 대한 가중치의 값이 다른 컬럼에 비해 높은 경우, 중요한 컬럼임
### - 가중치의 절대값이 클수록 영향력이 높은 특성임
print(X.info())
print(coef)


### LinearRegression 클래스는 학습 데이터를 예측하기 위해, 각 칼럼에 최적화된
### 가중치를 계산하는 머신러닝 클래스임. 학습 결과 나온 가중치는 학습 데이터에 
### 가장 맞는 가중치임. 머신러닝의 목표는 새로운 데이터를 예측하기 위해서인데, 
### 학습 데이터에만 맞는 가중치인 경우 과적합의 문제가 발생. 과적합의 문제는 
### L1 제약(Lasso) 또는 L2 제약(Ridge)을 둠으로써 해결할 수 있음.