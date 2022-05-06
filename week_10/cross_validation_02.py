# -*- coding: utf-8 -*-

##### 강의 주제: 교차 검증

### 1. 데이터 적재
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

### 2. 데이터 분할
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=0.3,
                                               stratify=y,
                                               random_state=1)

### 3. 데이터 전처리
from sklearn.preprocessing import StandardScaler

### 3-1. 스케일러 객체 생성
scaler = StandardScaler()

### 3-2. 스케일러 학습
scaler.fit(X_train)

### 3-3. 스케일 처리된 학습 데이터
X_train_scaled = scaler.transform(X_train)

### 3-4. 스케일 처리된 테스트 데이터
X_test_scaled = scaler.transform(X_test)

### 4. 머신러닝 모델 구축
from sklearn.linear_model import LogisticRegression

### 4-1. 머신러닝 모델 객체 생성
model = LogisticRegression(C=1.0,
                           n_jobs=-1,
                           random_state=1)

###
### 5. 교차 검증
###

### 구축한 머신러닝 모델을 사용하여 성능을 예측
### - 학습 진행 x
### - 교차 검증 수행을 통해 전체 데이터 셋에
###   대한 머신러닝 모델 성능을 예측

### cross_val_score
### - 교차 검증 기능을 수행
### - cross_val_score(예측기 객체, 
###                   전체 x 데이터, 
###                   전체 y 데이터, 
###                   교차 검증 개수)
### - 매개변수에 scoring 정보를 넣어 교차 검증 과정에서의 평가 방법을 수정할 수 있다.
### - 반환 값: '교차 검증 개수'만큼 '예측기 객체'가 생성되어 각 예측기의 평가 점수
###           가 반환된다. 예를 들어, 회귀 모델의 경우 R2 스코어가 반환되고 분류
###           모델의 경우 정확도가 반환된다. 

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model,
                            X_train_scaled,
                            y_train,
                            scoring='recall',
                            cv=5,
                            n_jobs=-1)

### 6. 교차검증 결과  평가
print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

### 7. 머신러닝 모델 학습
model.fit(X_train_scaled, y_train)

### 8. 머신러닝 모델 평가
score = model.score(X_train_scaled, y_train)
print(f'(MODEL) TRAIN SCORE : {score}')

score = model.score(X_test_scaled, y_test)
print(f'(MODEL) TEST SCORE : {score}')










