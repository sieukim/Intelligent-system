# -*- coding: utf-8 -*-

"""
Created on Fri May  20 15:10:21 2022

@author: 24
"""

##### 강의 주제: 파이프라인

### - 데이터 전처리를 함으로써 성능을 높일 수 있음
### - 이미 전처리 된 데이터를 활용하여 테스트를 하는 경우, 당연히 성능 좋음
### - 파이프라인을 활용하여 스케일러 -> 모델 순으로 처리 
### - KFold에 의해 K개로 나눈 후에, 하나는 예측 데이터로 남겨놓고
###   나머지는 학습데이터로 남겨놓음. 실제로는 k-1개의 폴드만 파이프에 들어감

### 1. 데이터 적재 및 설정  
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

print(X.shape)
print(y.shape)

### 2. 데이터 분할 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=11,
                                                    stratify=y)

### 3. 스케일러 객체 정의
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

### 4. 베이스 모델 정의
from sklearn.linear_model import LogisticRegression
base_model = LogisticRegression(n_jobs=-1, 
                                random_state=11,
                                l1_ratio=0)

### 5. 파이프라인 정의
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('s_scaler', scaler),
    ('base_model', base_model)])

### 6. 하이퍼 파라메터 정의
param_grid = [{'base_model__penalty': ['l2'],
               'base_model__solver': ['lbfgs'],
               'base_model__C': [1.0, 0.1, 10,  0.01, 100],
               'base_model__class_weight': ['balanced', {0: 0.9, 1: 0.1}]},
              {'base_model__penalty': ['elasticnet'],
               'base_model__solver': ['saga'],
               'base_model__C': [1.0, 0.1, 10, 0.01, 100],
               'base_model__class_weight': ['balanced', {0: 0.9, 1: 0.1}]}]

### 7. 교차 검증 데이터셋 분할
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=11)

### 8. 하이퍼 파라메터 검색 객체 정의 
### - 교차 검증 점수를 기반으로 최적의 하이퍼 파라메터를 검색 
### - GridSearchCV(
###     예측기 객체,
###     테스트에 사용할 하이퍼 파라미터 딕셔너리 객체,
###     cv=교차 검증 폴드 수,
###     ...)
from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(pipe, 
                          param_grid=param_grid,
                          cv=cv,
                          scoring='recall',
                          n_jobs=-1)

grid_model.fit(X_train, y_train)

### 9. 결과 확인
### - best_score_: 가장 높은 SCORE 값
### - best_params_: best_score_를 만든 하이퍼 파라메터 조합
### - best_estimator_: best_params_를 사용하여 생성된 모델 객체
print(f'best_score -> {grid_model.best_score_}')
print(f'best_params -> {grid_model.best_params_}')
print(f'best_model -> {grid_model.best_estimator_}')

### 10. 학습된 모델 평가 
print(f'SCORE(TRAIN): {grid_model.score(X_train, y_train)}')
print(f'SCORE(TEST): {grid_model.score(X_test, y_test)}')

from sklearn.metrics import classification_report
pred_train = grid_model.predict(X_train)
pred_test = grid_model.predict(X_test)
print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))

