# -*- coding: utf-8 -*-

## 강의 주제: 텐서플로우를 활용한 머신러닝 예제 

import tensorflow as tf

## 1. 학습 데이터 정의(회귀 분석용 데이터)
X_train = [10, 20, 30, 40, 50]
y_train = [5, 7, 15, 20, 25]


## 2. 데이터를 전달받기 위한 실행 매개변수 텐서 선언 
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])


## 3. 머신러닝을 위한 가설 정의
## - 선형 회귀 분석을 위한 선형 방정식
## - X * W + b

## 기울기(가중치) 변수 텐서 정의
w = tf.Variable(0, dtype=tf.float32)

## 절편(편향) 변수 텐서 정의
b = tf.Variable(0, dtype=tf.float32)

## 가설 정의
h = X * w + b


## 4. 학습 수행을 위한 오차 값 정의(MSE)
loss_1 = y - h
loss_2 = tf.square(loss_1)
loss = tf.reduce_mean(loss_2)


## 5. 오차 값을 감소시키는 방향으로 학습을 수행하는 객체를 정의
## - 학습률을 지정하여 학습 속도를 제어 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss) 


## 6. 변수 텐서 초기화 연산 텐서 정의
init_variables = tf.global_variables_initializer()


## 7. 세션 생성 및 텐서 실행, 세션 종료
with tf.compat.v1.Session() as sess :
    ## 변수 텐서 초기화
    sess.run(init_variables)
    
    ## feed_dict 정의
    feed_dict = {X: X_train, y: y_train}
    
    ## 학습 수행
    for step in range(1, 101):
        sess.run(train, feed_dict=feed_dict)
        
        ## 수행 상황 출력
        if step % 10 == 0:
            ## 예측 오차
            pred_loss = sess.run(loss, feed_dict=feed_dict)
            ## 가중치, 편향
            w_val, b_val = sess.run([w, b], feed_dict=feed_dict)
            
            print(f"step-{step}: 오차 {pred_loss} w {w_val} b {b_val}")            
    
    ## 최종 결과 출력 
    pred_loss = sess.run(loss, feed_dict=feed_dict)
    pred = sess.run(h, feed_dict=feed_dict)
    print(f"최종 오차: {pred_loss}")
    print(f"예측 결과: {pred}")
    

    from matplotlib import pyplot as plt
    
    ## 학습 데이터 시각화
    plt.plot(X_train, y_train, 'or')
    ## 예측 결과 시각화
    plt.plot(X_train, pred, '--b')
    
    ## 테스트 데이터 예측 결과 시각화 
    X_test = [37, 22]
    pred_test = sess.run(h, feed_dict={X : X_test})
    plt.plot(X_test, pred_test, 'xg')
    
    plt.show()
