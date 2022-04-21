# -*- coding: utf-8 -*-

###################################################################
### 과제                                                        
###                                                             
### - 당뇨 수치 데이터를 사용하여 앙상블 기반의 회귀분석 모델 구축   
### - 배깅과 그래디언트 부스팅을 사용                              
### - 모델을 구축한 후 학습 데이터와 테스트 데이터에 대한           
###   평균 절대 오차를 통해 모델의 적합성을 평가                                        
###################################################################


### 1. import pandas
import pandas as pd

### 2. import dataset
from sklearn.datasets import load_diabetes

### 3. load data
data = load_diabetes()

### 4. set data
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.Series(data=data.target)

### 5. import train_test_split
from sklearn.model_selection import train_test_split

### 6. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1)

###
### 7. BaggingRegressor
###

from sklearn.ensemble import BaggingRegressor

### 7-1. Create model using BaggingRegressor
model = BaggingRegressor(base_estimator=None, # None(Decision Regressor)
                         n_estimators=50,
                         max_samples=0.3,
                         max_features=0.3,
                         n_jobs=-1, 
                         random_state=1)

### 7-2. Fit 
model.fit(X_train, y_train)

### 7-3. Predict
pred = model.predict(X_train)

### 7-4. MSE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true=y_train, y_pred=pred)

print(f'Bagging Regressor\'s mean absolute error: {mae}')

### 7-5. Score
score = model.score(X_train, y_train)
print(f'Bagging Regressor Score(train): {score}')
score = model.score(X_test, y_test)
print(f'Bagging Regressor Score(test): {score}')

###
### 8. import GradientBoostingRegressor
###

from sklearn.ensemble import GradientBoostingRegressor

### 8-1. Create model using GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=50,
                                  subsample=0.3,
                                  max_features=0.3,
                                  random_state=1)

### 8-2. Fit 
model.fit(X_train, y_train)

### 8-3. Predict
pred = model.predict(X_train)

### 8-4. MSE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true=y_train, y_pred=pred)

print(f'Gradient Boosting Regressor\'s mean absolute error: {mae}')

### 8-5. Score
score = model.score(X_train, y_train)
print(f'Gradient Boosting Regressor\'s Score(train): {score}')
score = model.score(X_test, y_test)
print(f'Gradient Boosting Regressor\'s Score(test): {score}')

