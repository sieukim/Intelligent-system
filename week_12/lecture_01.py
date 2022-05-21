# -*- coding: utf-8 -*-

"""
Created on Fri May  20 15:10:21 2022

@author: 24
"""

##### 강의 주제: 하이퍼 파라메터 탐색 

### - train_test_split 함수는 train 데이터와 test 데이터로 분할함
### - train_test_split 함수는 random_state 설정을 통해 결과 조작이 가능함
### => train_test_split 함수를 이용하여 검증 데이터를 생성할 경우 조작의
###    여지가 있으므로, 교차 검증을 하여 조작의 여지를 없애야 함
###    KFold와 cross_val_score를 활용하여 교차 검증을 할 수 있음  

### 1. 데이터 적재 및 설정  
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

### 2. 데이터 분할 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)

### 3. 교차 검증 데이터셋 분할
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=1)

### 4. 교차 검증에 사용할 베이스 모델 정의
### - 테스트에 사용할 하이퍼 파라메터는 설정에서 제외
### - 공통으로 사용할 하이퍼 파라메터만 사용하여 정의 
from sklearn.ensemble import GradientBoostingClassifier
base_model = GradientBoostingClassifier(random_state=1)

### 5. 하이퍼 파라메터 정의
param_grid = {
        'learning_rate': [0.1, 0.2, 0.3, 1., 0.01],
        'max_depth': [1, 2, 3],
        'n_estimators': [100, 200, 300, 10, 50],
    }

### 6. 하이퍼 파라메터 검색 객체 정의 
### - 교차 검증 점수를 기반으로 최적의 하이퍼 파라메터를 검색 
### - GridSearchCV(
###     예측기 객체,
###     테스트에 사용할 하이퍼 파라미터 딕셔너리 객체,
###     cv=교차 검증 폴드 수,
###     ...)
from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(estimator=base_model, 
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=1,
                          verbose=3)

grid_model.fit(X_train, y_train)

### 7. 결과 확인
### - best_score_: 가장 높은 SCORE 값
### - best_params_: best_score_를 만든 하이퍼 파라메터 조합
### - best_estimator_: best_params_를 사용하여 생성된 모델 객체
print(f'best_score -> {grid_model.best_score_}')
print(f'best_params -> {grid_model.best_params_}')
print(f'best_model -> {grid_model.best_estimator_}')

### 8. 학습된 모델 평가 
print(f'SCORE(TRAIN): {grid_model.score(X_train, y_train)}')
print(f'SCORE(TEST): {grid_model.score(X_test, y_test)}')