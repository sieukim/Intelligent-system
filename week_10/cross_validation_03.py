# -*- coding: utf-8 -*-

### 강의 주제: 교차 검증 with kfold

### 1. 데이터 적재
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

### 2. 회귀 모델 객체 생성 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0,
                           class_weight='balanced',
                           n_jobs=-1,
                           random_state=1)

###
### 3. 교차 검증 
###

### cross_val_score 함수 매개변수로 전달된 예측기 객체 타입
### 1) 분류: 데이터 셔플 과정 선행
### 2) 회귀: 데이터 셔플 과정 생략

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, # 회귀 모델
                            X, y,
                            cv=3, 
                            scoring='accuracy', 
                            n_jobs=-1)

print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

### 4. 전체 데이터에 대한 사전 학습
model.fit(X, y)

### 5. 전체 데이터에 대한 사전 평가
score = model.score(X, y)
print(f'(MODEL) TRAIN SCORE : {score}')

score = model.score(X, y)
print(f'(MODEL) TEST SCORE : {score}')

###
### 6. 교차 검증
### 

### KFold 클래스
### - 교차 검증을 위해 사용
### - 매개변수 n_splits에 지정한 크기만큼 데이터를 분할하는
###   기능을 제공(기본적으론 셔플을 생략하고 순차적으로 분할함)
### - KFold 객체를 cross_val_score 함수의 매개변수 cv
###   로 전달할 수 있음 
### - 매개변수 shuffle에 True를 지정하는 경우 정답 데이터(y)
###   의 비율을 균등하게 포함하는 폴드 생성
### - 매개변수 shuffle을 지정하지 않는 경우 기본적으로 셔플을
###   생략하고 데이터를 순차적으로 분할 => 라벨이 정렬된 데이터
###   에는 잘못된 분석 결과가 나올 수 있음 

from sklearn.model_selection import KFold

### 6-1. KFold 객체 생성(shuffle 지정 x)
cv = KFold(n_splits=3, 
           random_state=11)

### 6-2. 생성한 KFold 객체를 활용하여 cross_val_score 객체 생성
cv_scores = cross_val_score(model, 
                            X, y,
                            cv=cv, 
                            scoring='accuracy', 
                            n_jobs=-1)

print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

### 6-3. KFold 객체 생성(shuffle 지정 o)
cv = KFold(n_splits=3, 
           shuffle=True, 
           random_state=11)

### 6-4. 생성한 KFold 객체를 활용하여 cross_val_score 객체 생성
cv_scores = cross_val_score(model, 
                            X, y,
                            cv=cv, 
                            scoring='accuracy', 
                            n_jobs=-1)

print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

















