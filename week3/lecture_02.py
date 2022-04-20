# -*- coding: utf-8 -*-

import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

### 1. 빈 데이터 프레임 객체 생성(설명 변수 생성)
X = pd.DataFrame()

## 1-1. rate 컬럼 추가
X['rate'] = [0.3, 0.8, 0.0999]

## 1-2. price 컬럼 추가
X['price'] = [10000, 5000, 9500]

### 2. 종속 변수 생성
y = pd.Series([0, 1, 0])

# 설명 변수, 종속 변수 출력
print(f'Initial X: {X}')
print(f'Initial y: {y}')

### 3. 스케일 전처리
###    : price 컬럼의 값을 rate 컬럼의 값과 동일한 범위를 갖도록 스케일 전처리 수행
###      (데이터 값은 수정되지만, 원본 값에서 갖는 상대적인 크기는 유지함)
from sklearn.preprocessing import MinMaxScaler

## 3-1.  MinMaxScaler 객체 생성
scaler = MinMaxScaler()

## 3-2. MinMaxScaler 학습
##      (
##       1단계: 각 컬럼 별 최소값과 최대값 추출,
##       2단계: 각 컬럼 별 (원본값 - 최소값) / (최대값 - 최소값) 연산을 수행하여 데이터 변환
##      )
scaler.fit(X)

## 3-3. 스케일 전처리 수행
X = scaler.transform(X)

print(f'After preprocessing X: {X}')

### 4. 최근접 이웃 알고리즘 수행
from sklearn.neighbors import KNeighborsClassifier

## 4-1. 최근접 이웃 알고리즘 객체 생성
model = KNeighborsClassifier(n_neighbors=1)

## 4-2. 최근접 이웃 알고리즘 학습
model.fit(X, y)

## 4-3. 예측 데이터 생성
X_pred = [[0.81, 7000]]

## 4-4. 예측 데이터 스케일 전처리 수행
X_pred = scaler.transform(X_pred)

## 4-5. 예측 수행
pred = model.predict(X_pred)
print(pred)



### 최근접 이웃 알고리즘의 학습 및 예측 방법
### - 학습: fit 메소드에 입력된 데이터를 단순 저장함
### - 예측: fit 메소드에 의해 저장된 데이터와 예측하고자 하는 데이터간의 유클리드 거리
###        를 계산하여, 가장 인접한 이웃을 추출. 추출된 이웃의 클래스를 사용하여 다수
###        결 과정을 수행