# -*- coding: utf-8 -*-
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 회귀 분석 with 제약

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


###
### 3. 데이터 분할
###

from sklearn.model_selection import train_test_split

### 3-1. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1)


###
### 4. 머신러닝 모델 구축
###

## LinearRegression
from sklearn.linear_model import LinearRegression

## L1 제약(Lasso)
## - 모든 컬럼 중 특정 컬럼에 대해서만 가중치의 값을 할당하는 제약 조건 (다른 컬럼의 
##   가중치 값은 0으로 제약)
## - 컬럼이 많은 데이터를 학습하는 경우 빠른 학습 속도를 보임
## - 모든 컬럼 중 중요도가 높은 컬럼을 구분할 수 있음
from sklearn.linear_model import Lasso

## L2 제약(Ridge)
## - 모든 컬럼의 가중치 값을 0으로 수렴하게 제어하는 제약 조건
## - LinearRegression은 학습 데이터에 최적화하도록 학습하기 때문에, 테스트 데이터에
##   대한 일반화 성능이 감소됨
## - 일반화 성능을 올리기 위해 모든 컬럼을 적절히 활용할 수 있도록 사용하는 제약 조건
from sklearn.linear_model import Ridge

### 4-1. Linear Regression 모델 객체 생성
### n_jobs가 -1인 경우 사용할 수 있는 n_jobs를 모두 사용
model = LinearRegression(n_jobs=-1)

### Lasso와 Ridge의 하이퍼 파라미터 alpha
### - alpha 값이 커질수록 제약을 크게 설정, 모든 칼럼의 가중치 값이 0으로 수렴
### - alpha 값이 작아질수록 제약을 작게 설정, 모든 칼럼의 가중치 값이 0에서 발산
### - alpha 값이 작아질수록 LinearRegression과 동일해짐

### 4-2. Lasso 모델 객체 생성
lasso_model = Lasso(alpha=0.001, random_state=1)

### 4-3. Ridge 모델 객체 생성
ridge_model = Ridge(alpha=10000.0, random_state=1)

### 4-4. 머신러닝 모델 객체 학습
model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

### 4-5. 가중치 확인
print(f'가중치(LR): {model.coef_}')
print(f'가중치(L1): {lasso_model.coef_}')
print(f'가중치(L2): {ridge_model.coef_}')

### 4-6. 머신러닝 모델 객체 평가

## 학습 데이터 사용
score = model.score(X_train, y_train)
print(f'Train(LR): {score}')

score = lasso_model.score(X_train, y_train)
print(f'Train(L1): {score}')

score = ridge_model.score(X_train, y_train)
print(f'Train(L2): {score}')

## 테스트 데이터 사용
score = model.score(X_test, y_test)
print(f'Test(LR): {score}')

score = lasso_model.score(X_test, y_test)
print(f'Test(L1: {score}')

score = ridge_model.score(X_test, y_test)
print(f'Test(L2): {score}')