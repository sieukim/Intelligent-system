# -*- coding: utf-8 -*-

## 강의 주제: 회귀 분석 with 텐서플로우 

import tensorflow as tf

## 1. 학습 데이터 정의
X_train = [10., 20, 30, 40, 50]
y_train = [5., 7, 15, 20, 25]


## 2. 테스트 데이터 정의
X_test = [60., 70, 80]
y_test = [32., 38, 40]


## 3. 데이터를 전달받기 위한 실행 매개변수 텐서 정의 
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])


## 4. 학습 과정에 갱신되는 변수 텐서 정의
## - 가중치와 절편 변수 텐서 선언
w = tf.Variable(0.0)
b = tf.Variable(0.0)
 

## 5. 가설 텐서 정의 
## - 사이킷런의 예측기 클래스가 제공하는 predict 메소드의 값을 생성하는 식
h = X * w + b


## 6. 오차 값을 계산하는 연산 텐서 정의
## - tf.square()를 활용하여 평균 제곱 오차를 계산
## - tf.reduce_mean()을 활용하여 오차 값의 평균을 계산
loss = tf.reduce_mean(tf.square(y - h))   


## 7. 경사하강법 객체 선언

## tf.train.GradientDescentOptimizer 클래스
## - learning_rate에 지정된 값(기본=0.01)을 사용하여 학습을 수행하는 클래스
## - 변수 텐서의 값을 조정하여 minimize 메소드의 매개변수로 전달된 텐서의 
##   값이 작아지도록 조정 
## - 적절한 learning_rate를 찾아 테스트해야 함 

## tf.train.AdamOptimizer 클래스
## - learning_rate를 스스로 조정하여 최적의 
##   learning_rate를 찾아 학습을 수행하는 클래스 
optimizer = tf.compat.v1.train.AdamOptimizer()

## 경사하강법 객체를 사용하여 loss 값이
## 줄어들 수 있도록 학습을 수행할 텐서를 생성
train = optimizer.minimize(loss)


## 8. 세션 생성, 학습 수행, 세션 종료
with tf.compat.v1.Session() as sess:
    ## 변수 텐서 초기화
    sess.run(tf.global_variables_initializer())
 
    ## fead_dict 정의
    feed_dict = {X: X_train, y: y_train}
    
    ## 학습 수행
    for step in range(1, 100001) :
        sess.run(train, feed_dict=feed_dict)
        
        if step % 100 == 0:
            w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict=feed_dict)
            print(f"step-{step}: 오차 {loss_val} w {w_val} b {b_val}")      
    
    ## 학습 결과 출력
    print('학습 종료')
    print(f"오차 {loss_val} w {w_val} b {b_val}")     
    
    ## 테스트 결과 출력
    feed_dict = {X: X_test, y: y_test}
    pred, loss_val = sess.run([h, loss], feed_dict=feed_dict)
    print('테스트 결과')
    print('실제 정답 : ', y_test)
    print('예측 : ', pred)
    print('오차 : ', loss_val)
            
    from matplotlib import pyplot as plt
    
    ## 학습 데이터, 학습 데이터 예측 결과 시각화
    feed_dict = {X: X_train, y: y_train}
    plt.scatter(X_train, y_train)
    plt.plot(X_train, sess.run(h, feed_dict=feed_dict))
    plt.show()
    
    ## 테스트 데이터, 테스트 데이터 예측 결과 시각화 
    feed_dict={X:X_test, y:y_test}
    plt.scatter(X_test, y_test)
    plt.plot(X_test, sess.run(h, feed_dict=feed_dict))
    plt.show()
