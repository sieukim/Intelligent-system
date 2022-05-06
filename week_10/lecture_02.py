# -*- coding: utf-8 -*-

import pandas as pd

##### 강의 주제: 데이터 전처리 with user-defined data

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

### 1. 데이터 정의
X = pd.DataFrame()

X['gender'] = ['F', 'M', 'F', 'F', 'M']
X['age'] = [15, 35, 25, 37, 55]

print(X)
print(X.info())

###
### 2. 데이터 전처리 
###

### 2.1 문자열 데이터 열 리스트와 수치형 데이터 열 리스트
obj_columns=['gender']
num_columns=['age']

### 2.2 문자열 데이터 전처리 클래스 로딩
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

### 2.3 수치형 데이터 전처리 클래스 로딩
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()        

### 2.4 전처리를 돕는 클래스 로딩
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('scaler', scaler, num_columns),
     ('encoder', encoder, obj_columns)])

### 전처리 학습
ct.fit(X)

### 전처리 수행
X = ct.transform(X)