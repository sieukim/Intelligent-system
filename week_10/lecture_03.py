# -*- coding: utf-8 -*-

import pandas as pd

##### 강의 주제: 데이터 전처리 - 결측 데이터 처리

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

X['gender'] = ['F', 'M', 'F', 'F', None] # 결측 데이터 발생 
X['age'] = [15, 35, 25, 37, 55]

print(X)
print(X.info())
print(X.isnull().sum()) # 결측 데이터 확인

### 2. 데이터 전처리

### 2.1 문자열 데이터 열 리스트와 수치형 데이터 열 리스트
obj_columns=['gender']
num_columns=['age']

### 2.2 문자열 데이터 전처리 클래스 로딩
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

### 2.3 수치형 데이터 전처리 클래스 로딩
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()        

###
### 2.4 결측 데이터 전처리 클래스 로딩
###

### - 결측 데이터를 처리하는 클래스 
from sklearn.impute import SimpleImputer

### missing_values: 결측 데이터 정의
### strategy: 결측 데이터 대체 방법

### 심플 임퓨터 모델 객체 생성
obj_imputer = SimpleImputer(missing_values=None, strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

###
### 2.5 반복적인 작업을 쉽게 할 수 있게 하는 클래스 로딩
###

### steps: 작업 순서 리스트 

from sklearn.pipeline import Pipeline

num_pipe = Pipeline(
    steps=[('imputer_num', num_imputer), # 결측 데이터 처리
           ('scaler', scaler)])          # 스케일 처리

obj_pipe = Pipeline(
    steps=[('imputer_obj', obj_imputer), # 결측 데이터 처리
           ('encoder', encoder)])        # 인코딩 


### 2.4 전처리를 돕는 클래스 로딩
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('num_pipe', num_pipe, num_columns),
     ('obj_pipe', obj_pipe, obj_columns)])

### 전처리 학습
ct.fit(X)

### 전처리 수행
X = ct.transform(X)
