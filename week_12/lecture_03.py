# -*- coding: utf-8 -*-

import pandas as pd

### 1. 데이터 적재
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

### 2. 데이터 설정 
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

## EDA
print(X.head())
print(X.info())
print(X.describe())
print(y.head())
print(y.describe())

### 3. 데이터 분할 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1)

### 4. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

## 스케일러 객체 정의
s_mm = MinMaxScaler()
s_ss = StandardScaler()
s_rs = RobustScaler()

## 스케일러 적용 컬럼 정의
mm_cols = ['MedInc', 
           'HouseAge', 
           'AveRooms', 
           'AveBedrms']
ss_cols = ['AveOccup', 
           'Latitude', 
           'Longitude']
rs_cols = ['Population']

## 컬럼 트랜스포머 객체 생성 
## (스케일러 이름, 스케일러 객체, 적용 컬럼)
from sklearn.compose import ColumnTransformer
pp = ColumnTransformer(
    [("s_mm", s_mm, mm_cols),
     ("s_ss", s_ss, ss_cols),
     ("s_rs", s_rs, rs_cols)], 
     n_jobs=-1)

### 5. 모델 객체 생성 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1,
                              random_state=1)

### 6. 파이프라인 객체 생성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('pp', pp), ['model', model]])

### 7. 교차 검증 데이터셋 분할
from sklearn.model_selection import KFold
cv = KFold(n_splits=15, shuffle=True, random_state=1)

### 8. 하이퍼 파라미터 정의 
param_grid = {'model__n_estimators': [100, 50, 20, 200, 300],
              'model__max_depth': [None, 5, 7, 9, 13],
              'model__max_samples': [None, 0.5, 0.3, 0.7]}


### 9. 하이퍼 파라메터 검색 객체 정의 
from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(pipe, 
                          param_grid=param_grid,
                          cv=cv,
                          scoring='r2',
                          verbose=3,
                          n_jobs=-1)

grid_model.fit(X_train, y_train)

### 10. 학습된 모델 평가 
print(f'SCORE(TRAIN): {grid_model.score(X_train, y_train)}')
print(f'SCORE(TEST): {grid_model.score(X_test, y_test)}')