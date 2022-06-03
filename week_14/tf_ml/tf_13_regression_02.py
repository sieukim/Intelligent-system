# -*- coding: utf-8 -*-

## 강의 주제: 회귀 분석 with 텐서플로우
##          (X 데이터의 특성이 다수개일 때, 
##          다수개의 가중치를 만들어 학습을 
##          수행하는 경우)

import numpy as np
import tensorflow as tf

## 1. 학습 데이터 정의
## - X: 키, 성별
## - y: 몸무게
X_train = np.array([
    [158, 1],
    [170, 1],
    [183, 1],
    [191, 1],
    [155, 0],
    [163, 0],
    [180, 0],
    [158, 0],
    [170, 0]])
y_train = np.array([64, 86, 84, 80, 49, 59, 67, 54, 67])


## 2. 학습 데이터를 전달받기 위한 실행 매개변수 텐서 선언
## - 다차원 배열을 실행 매개변수로 전달받는 경우,
##   행의 개수와 상관없이 특성이 2개인 데이터를 전달
X = tf.placeholder(
    tf.float32, shape=[None, 2])
y = tf.placeholder(
    tf.float32, shape=[None])


## 4. 학습 과정에 갱신되는 변수 텐서 정의
## - 가중치와 절편 변수 텐서 정의
## - 각 특성에 대한 가중치 변수 텐서 정의
w1 = tf.Variable(1.0)   # 키에 대한 가중치
w2 = tf.Variable(1.0)   # 성별에 대한 가중치
b = tf.Variable(0.0)


## 5. 가설 텐서 정의 
h = X[:,0] * w1 + X[:,1] * w2 + b


## 6. 오차 값을 계산하는 연산 텐서 정의
loss = tf.reduce_mean(tf.square(y - h))


## 7. 경사하강법 객체 선언
optimizer = tf.compat.v1.train.AdamOptimizer()
train = optimizer.minimize(loss)


## 8. 세션 생성, 학습, 세션 종료 
with tf.compat.v1.Session() as sess :
    ## 변수 텐서 초기화
    sess.run(tf.global_variables_initializer())
 
    ## fead_dict 정의
    feed_dict={X:X_train, y:y_train}    
    
    ## 학습 횟수를 지정하지 않고, 
    ## 오차의 값을 기준으로 하여 학습을 수행
    step = 1
    prev_loss = None
    
    while True :    
        sess.run(train, feed_dict=feed_dict)        
        
        if step % 100 == 0 :
            w1_val, w2_val, b_val, loss_val = sess.run([w1, w2, b, loss], feed_dict=feed_dict)
            print(f"step-{step}: 오차 {loss_val} w1 {w1_val} w2 {w2_val} b {b_val}")
        
        ## 현재 오차
        current_loss = sess.run(loss, feed_dict=feed_dict)
        
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

    ## 학습 결과 확인 
    from sklearn.metrics import r2_score
    pred = sess.run(h, feed_dict=feed_dict)
    print("r2 점수 : ", r2_score(y_train, pred))    
    print("실제 정답 : ", y_train)
    print("예측 : ", pred)
    
    
    
    
            
    





















