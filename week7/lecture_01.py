# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 02:02:02 2022

@author: potato
"""

import numpy as np
import pandas as pd

## 출력 컬럼의 개수 제어
pd.options.display.max_columns = 30
## 출력 로우의 개수 제어
pd.options.display.max_rows = 30

##### 강의 주제: 데이터 전처리 with titanic

### 1. 데이터 적재

### 1-1. 데이터 파일
file_name = './titanic.csv'

### 1-2. csv 파일 내 데이터 읽기
data = pd.read_csv(file_name, 
                   header='infer', 
                   sep=',')

### 1-3. 데이터 확인
data.head() # => 데이터 확인 결과 문자열 존재

### 1-4. 데이터 EDA
print(data.info())


###
### 2. unique한 데이터로 만들 수 있는 칼럼 제거 (ex. Id...)
###

### - DataFrame의 drop 메소드를 사용
### - drop(columns=제거할 column 리스트)
data2 = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], 
                  inplace=False)

### 2-1. 데이터 EDA
print(data2.info())

### 2-2. 데이터 내 결측 데이터 확인
print(data2.isnull().sum())  # => Age와 Embarked에 결측 데이터 존재


###
### 3. 결측 데이터 제거
###

### 결측 데이터 처리 방법
### 
### 1. 결측 데이터를 포함하는 칼럼을 제거
### - 칼럼 자체를 지우는건 데이터 개수가 많아 무리가 있음
###
### 2. 결측 데이터를 포함하는 로우를 제거
###
### 3. 결측 데이터를 기초 통계 데이터로 대체
### - 수치형 데이터: 평균, 중심값, 최빈값 등을 사용
### - 문자열 데이터: 최빈값을 사용
###
### 4. 결측 데이터를 지도학습 기반 머신러닝 모델을 구축하여 예측한 값으로 대체
### - Age 컬럼에 적합
###
### 5. 결측 데이터를 준지도학습, 비지도학습 기반 머신러닝 모델을 구축하여 
###    예측한 값으로 대체
### - Cabin 컬럼에 적합
###    

### 3-1. 결측 데이터를 갖는 컬럼 제거
data3 = data2.dropna(subset=['Age', 'Embarked'])

### 3-2. 결측 데이터를 갖는 컬럼을 제거한 데이터 EDA
print(data3.info())


###
### 4. 문자열 데이터 전처리
###

### 문자열 데이터 전처리 필요성: 머신러닝 알고리즘은 문자열을 처리하지 않음
### 따라서 문자열 데이터를 터를 수치형 데이터로 변환해야 함

### 문자열 데이터 전처리 위치: 데이터 분할 전에 하는 것이 원칙

### 문자열 데이터 전처리 방식: 기본적으로는 문자열 데이터를 수치 데이터로 연계하여
### 변환하는 매칭 방법을 사용하여 처리
###
### 1. 라벨인코딩: 특정 문자열 데이터를 정수와 매칭하여 단순 변환하는 방식
### - ex) 남성/여성 -> 0/1
### -     S/Q/C -> 0/1/2
### - 일반적으로 정답 데이터가 문자열 데이터로 구성된 경우 사용
### - 설명변수에는 잘 사용하지 않음
### 
### 2. 원핫인코딩: 문자열 종류의 개수만큼 컬럼을 생성하고, 그 중 하나의 위치에만
### 1을 대입하는 방식
### - ex) 남성/여성 -> 10/01
### -     S/Q/C -> 100/010/001
### - 일반적으로 설명변수가 문자열 데이터로 구성된 경우 사용
### - 메모리 낭비가 심하므로 문자열 종류를 줄여 접근하는 것이 좋음

### 4-1. 데이터 설정
### 문자열 데이터 전처리는 수치형 데이터 전처리와 과정이 독립적이므로 데이터를 분할
### 하는 것이 좋음

X = data3.iloc[:, 1:]
y = data3.Survived

### 4-2 수치형 데이터를 갖는 칼럼명 리스트
X_num = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

### 4-3. 문자열 데이터를 갖는 칼럼명 리스트
X_obj = [cname for cname in X.columns if X[cname].dtype not in ['int64', 'float64']]

### 4-4. 수치형 데이터
X_num = X[X_num]

### 4-5. 문자열 데이터
X_obj = X[X_obj]

### 4-6. 문자열 데이터 원핫인코딩
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False,            # 희소행렬 생성 여부
                        handle_unknown='ignore') # 학습 과정에서 인지하지 못한 문자열 값에 대한 처리 프로세스

### OneHotEncoder의 메소드
### 1. fit: 전처리 과정에 필요한 정보 수집
### 2. transform: 수집한 정보를 바탕으로 데이터를 변환
### 3. fit_transform: fit과 transform을 동시에

X_obj_encoded = encoder.fit_transform(X_obj)    

print(X_obj.head())
print(X_obj_encoded[:5])

### preprocessing 내 모든 클래스의 transform 메소드는 넘파이 배열을 반환
### => 수치형 데이터와 문자열 데이터를 합쳐 모델 학습에 사용하기 전, 
###    전처리된 문자열 데이터를 판다스 데이터프레임으로 변환하는 작업 필요
X_obj_encoded = pd.DataFrame(data=X_obj_encoded, 
                             columns=['s_f','s_m',
                                      'e_C','e_Q','e_S'])

print(X_obj_encoded.head())

### 4-7. 설명변수 생성
X_num.reset_index(inplace=True)
X_obj_encoded.reset_index(inplace=True)

X = pd.concat(objs=[X_num, X_obj_encoded],
              axis=1)

### 4-8. 설명변수 EDA
print(X.info())


### 5. 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)


### 6. 머신러닝 모델 구축
from sklearn.ensemble import RandomForestClassifier

### 6-1. 머신러닝 모델 객체 생성
model = RandomForestClassifier(n_estimators=100,
                               max_depth=None,
                               max_samples=1.0,
                               class_weight='balanced',
                               n_jobs=-1,
                               random_state=1)

### 6-2. 머신러닝 모델 객체 학습
model.fit(X_train, y_train)

### 6-3. 머신러닝 모델 객체 평가
score = model.score(X_train, y_train)
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')
