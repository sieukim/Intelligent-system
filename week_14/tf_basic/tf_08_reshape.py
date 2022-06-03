# -*- coding: utf-8 -*-

## 강의 주제 - 형태(shape) 변경 

import tensorflow as tf

## tf.reshape(tensor, shape)
## - 매개변수로 전달받은 tensor의 형태를
##   매개변수로 전달받은 shape로 변경한
##   텐서를 반환 
## - -1 사용 가능 

## tf.zeros(shape)
## - 주어진 shape의 형태를 가지며
##   모든 요소가 0인 텐서를 반환 

## 1. 변수 텐서 정의 
X = tf.Variable(tf.zeros([10]))     

## 2. 형태 변경 
X_reshape = tf.reshape(X, [-1, 2])  

## 3. 변수 텐서 초기화 연산 텐서 정의 
init_variables = tf.global_variables_initializer()

## 4. 세션 생성 및 텐서 실행, 세션 종료 
with tf.compat.v1.Session() as sess :
    ## 변수 텐서 초기화
    sess.run(init_variables)
    
    result = sess.run(X)
    print(f"result = {result}") # result = [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    result = sess.run(X_reshape)
    print(f"result = {result}") # result = [[0. 0.] [0. 0.] [0. 0.] [0. 0.] [0. 0.]]
