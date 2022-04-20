# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:10:50 2022

@author: potato
"""

import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 데이터 분석 수행 과정


###
### 1. 데이터 적재(로딩)
###

## 유방암 데이터셋(분류용)
from sklearn.datasets import load_breast_cancer

## 데이터 적재
data = load_breast_cancer()

# 데이터 키 출력
print(data.keys())

## X - 특정 종속 변수를 유추하기 위해서 정의된 데이터셋, 설명변수
X = pd.DataFrame(data=data.data, columns=data.feature_names)
## y - 정답 데이터(레이블)
y = pd.Series(data=data.target)


###
### 2. 데이터 관찰(탐색)
###

## 설명변수 X 관찰 
## : 데이터 개수, 컬럼 개수, 결측 데이터 존재 유무, 데이터 타입
## : 수치형 데이터만 활용 가능
print(X.info())

## 각 컬럼에 대한 기초 통계 정보 확인
## : 데이터 개수(count), 평균(mean), 표준편차(std), 
##   최솟값(min), 최댓값(max), 4분위 수(25%, 50%, 75%)
## : 스케일(값의 범위)를 중점적으로 체크, 
##   각 컬럼 별 스케일의 오차가 발생하는 경우 스케일 전처리 필요
print(X.describe())

## 종속변수 y 관찰
print(y)
## 범주형 종속 변수인 경우 범주형 값과 개수 확인
print(y.value_counts())
## 범주형 종속 변수인 경우 값의 개수 비율이 중요함
## => 극단적인 케이스로 악성 1% 정상 99%인 경우,
##    정상으로만 예측해도 99%의 정확도를 가짐
## => 데이터의 비율 차가 큰 경우 적은 데이터의 양을 늘리는 '오버 샘플링'
##    또는 많은 데이터의 양을 줄이는 '언더 샘플링'이 필요 
print(y.value_counts() / len(y))

## 데이터 전처리는 학습 데이터에 대해서 수행함
## 테스트 데이터는 학습 데이터를 반영한 결과로 수행함

## + 데이터 스케일 처리(나중에 다시 배움, 참고용)
## MinMaxScaler: 스케일을 조정하는 정규화 함수
##               모든 데이터가 0과 1 사이의 값을 갖도록 함
from sklearn.preprocessing import MinMaxScaler

## 스케일러 인스턴스 생성
minMaxScaler = MinMaxScaler()

## 설명변수 X의 전체 데이터를 학습
minMaxScaler.fit(X)

## 모든 데이터를 0과 1 사이의 값으로 변환
X = minMaxScaler.transform(X)

print(X)

## 위 전처리 코드는 데이터의 전체 모습을 미리 확인한 수
## 데이터가 분할되는 결과를 가지기 때문에 학습 성능은 
## 올라가지만 실전에서 망가질 수 있음 


###
### 3. 데이터 분할
###

## 머신러닝 - 학습데이터:테스트데이터 = 8:2
## 딥러닝 - 학습데이터:검증데이터:테스트데이터 = 6:2:2
## (딥러닝은 부분 배치 학습을 수행하여 점진적으로 학습량을 늘려가는 경우가 많음. 
## 중간 점검의 의미로 검증 데이터를 활용)

## 데이터 분할
## train_test_split: 데이터 분할 함수
from sklearn.model_selection import train_test_split
## train_test_split(X=설명변수, 
##                  y=종속변수,
##                  test_size=테스트데이터 비율,
##                  train_size=학습데이터 비율,
##                  stratify=범주형 데이터를 다룰 때 중요한 옵션으로 종속변수를 넣어
##                           종속 변수의 클래스 비율을 유지시킴,
##                  random_state=데이터 분할이 이루어질 때 발생하는 셔플의 시드값으로,
##                               데이터의 분할된 값이 항상 동일한 값을 유지하도록 함
##                  )
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    train_size=0.8,
                                                    stratify=y,
                                                    random_state=50)

## 분할된 데이터의 개수 확인
print(len(X_train), len(X_test))
print(len(y_train), len(y_test))


###
### 4. 데이터 전처리
###

## 스케일 처리(MinMax, Standart, Robust...)
## 인코딩 처리(label encoding, one-hot encoding)
## 차원 축소
## 특성 공학 
## ...

## MinMaxScaler 인스턴스 생성
minMaxScaler = MinMaxScaler()

## 전처리 과정의 데이터 학습은 학습 데이터를 기준으로 수행
minMaxScaler.fit(X_train)

## 학습 데이터: 스케일 전처리 수행
X_train = minMaxScaler.transform(X_train)
## 테스트 데이터: 학습 데이터를 기준으로 변환 
X_test = minMaxScaler.transform(X_test)


###
### 5. 머신러닝 모델 구축
###

## KNeighborsClassifier: 최근접 이웃 알고리즘
## - 분류를 알 수 없는 데이터에 대하여 K개의 이웃
##   데이터의 분류를 확인한 후 다수결에 의해 결정
## - K는 하이퍼파라미터로 적절한 설정이 중요
## - K가 너무 작은 경우엔 과적합, 너무 큰 경우엔 과소적합이 발생할 수 있음
## - K는 n_neighbors를 이용하여 설정
from sklearn.neighbors import KNeighborsClassifier

## 머신러닝 모델 객체 생성
model = KNeighborsClassifier(n_neighbors=10)

## 머신러닝 모델 객체 학습
## - fit 메소드 사용
## - +) 사이킷런의 모든 머신러닝 클래스는 fit 메소드를 이용함. 
##      매개변수로는 X와 y를 받으며, X는 반드시 2차원 데이터셋(DataFrame)
##      이어야 하며 y는 반드시 1차원 데이터셋(Series, list...)이어야 함
model.fit(X_train, y_train)

## 머신러닝 모델 객체 평가
## - score 메소드 사용
## - 입력된 X를 사용하여 예측을 수행하고,
##   예측된 값을 입력된 y와 비교하여,
##   평가 결과를 반환
score = model.score(X_train, y_train)
print(f'Train: {score}')

score = model.score(X_test, y_test)
print(f'Test: {score}')

## 주의 사항!
## - 머신러닝 클래스 타입이 분류형인 경우 score 메소드는 정확도를 반환
##   정확도: 전체 데이터에서 정답 데이터의 비율
## - 머신러닝 클래스 타입이 회귀형인 경우 score 메소드는 결정계수를 반환
##   결정계수: 음수부터 시작하여 1까지의 값을 가지며, 1은 100%를 의미함

## 머신러닝 모델 객체 예측
## - predict 메소드 사용
## - 예측할 데이터 X는 반드시 2차원 데이터셋(DataFrame)이어야 함
pred = model.predict(X_train[:10])
print(pred)
print(y_train[:10])

## 분류형 머신러닝 모델인 경우
## - 확률 값으로 예측 가능 (일부 클래스에선 제공하지 않음)
## - predict_proba 메소드 사용
## - 예측할 데이터 X는 반드시 2차원 데이터셋(DataFrame)이어야 함
proba = model.predict_proba(X_test[:10])
print(f'Proba: {proba}')
