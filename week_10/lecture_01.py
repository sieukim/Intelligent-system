# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:10:21 2022

@author: 24
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 데이터 전처리 with breast cancer dataset

### 데이터 전처리
##
### 1. 문자열 데이터
### - 결측 데이터 처리
### - 라벨 인코딩
### - 원핫 인코딩
##
### 2. 수치형 데이터 
### - 결측 데이터 처리
### - 이상치 제거(대치)
### - 스케일 처리

### 1. 데이터 적재
data = load_breast_cancer()

### 2. 데이터 설정
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# EDA
print(X.info())
print(X.describe())

###
### 3. 데이터 전처리
###

# 전처리를 적용할 컬럼을 식별
num_columns = X.columns

### 3-1. 전처리 클래스 로딩
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

###
### 3-2. ColumnTransformer 클래스 로딩
### 

### - 열마다 다른 변환을 적용하도록 돕는 클래스
from sklearn.compose import ColumnTransformer

### transformers: transformer 리스트
### transformer: (이름, 변환기, 변환기를 적용할 열의 리스트)

### 컬럼 트랜스포머 모델 객체 생성
ct = ColumnTransformer(transformers=[('scaler', scaler, num_columns)])

### 컬럼 트랜스포머 모델 학습
ct.fit(X)

### 컬럼 트랜스포머 모델 결과(=전처리 결과)
X = ct.transform(X)
