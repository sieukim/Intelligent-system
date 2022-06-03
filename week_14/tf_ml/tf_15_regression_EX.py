# -*- coding: utf-8 -*-

## 강의 주제: 사이킷런의 당뇨병 데이터셋을 분석하여 결과를 확인
##          (텐서플로우로 학습했을 때 결과와 
##           사이킷런으로 학습했을 때 결과를 비교)

import pandas as pd
import tensorflow as tf

pd.options.display.max_columns = 100

## 1. 데이터셋 적재
from sklearn.datasets import load_diabetes
data = load_diabetes()


## 2. 설명변수, 종속변수 설정
X_df = pd.DataFrame(data.data)
y_df = pd.Series(data.target)

print(X_df.info())
print(X_df.describe())
print(y_df.describe())


## 3. 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y_df.values, random_state=1)
    

## 4. 데이터 입력을 위한 실행 매개변수 텐서 정의
X = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None])

## 5. 가중치(기울기), 절편의 텐서를 정의 
W = tf.Variable(tf.zeros(shape=[10, 1]))
b = tf.Variable(0.0)

## 6. 가설(예측식)
## - 오차를 계산하기 위해서 1차원 텐서로 형변환
h = tf.reshape(tf.matmul(X, W) + b, [-1])

## 7. 오차 계산(퍙균제곱오차)
loss = tf.reduce_mean(tf.square(y - h))

## 8. 손실을 감소시키는 방향으로 학습을 진행하기 위한 OPTIMIZER 객체 선언
optimizer = tf.compat.v1.train.AdamOptimizer()

## 9. OPTIMIZER 객체를 사용하여 학습을 진행시키기 위한 텐서 선언
train = optimizer.minimize(loss)

## 10. TensorFlow의 세션 객체를 생성하여 학습을 진행하고 결과를 확인
with tf.compat.v1.Session() as sess :
    # Variable 초기화
    sess.run(tf.global_variables_initializer())
    
    # feed_dict 정의 
    feed_dict = {X : X_train, y : y_train}
    
    step = 1
    prev_loss = None
    
    while True :
        sess.run(train, feed_dict=feed_dict)
        
        # 현재 오차
        current_loss = sess.run(loss, feed_dict=feed_dict)
        
        if step % 100 == 0 :  
            print(f"step-{step}: loss {current_loss}")
            print("{0} loss : {1}".format(step, current_loss))
        
        ## 이전 오차가 없는 경우
        if prev_loss == None:
            ## 오차 갱신
            prev_loss = current_loss
        ## 이전 오차보다 작은 경우
        elif prev_loss > current_loss:
            ## 오차 갱신
            prev_loss = current_loss
        ## 이전 오차보다 큰 경우
        else:
            ## 학습 종료
            break
            
        step += 1
        
    from sklearn.metrics import r2_score
        
    ## 학습 데이터에 대한 결과 확인
    feed_dict = {X : X_train}
    pred = sess.run(h, feed_dict=feed_dict)
    print("Tensorflow 학습 결과 : ", r2_score(y_train, pred))
   
    ## 테스트 데이터에 대한 결과 확인 
    feed_dict = {X : X_test}
    pred = sess.run(h, feed_dict=feed_dict)
    print("Tensorflow 테스트 결과 : ", r2_score(y_test, pred))

## 사이킷런 학습 결과 확인 
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
print("Scikit-learn 학습 결과 : ", model.score(X_train, y_train))
print("Scikit-learn 테스트 결과 : ", model.score(X_test, y_test))






















