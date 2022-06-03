# -*- coding: utf-8 -*-

## 강의 주제: 연산의 분배 법칙 

import numpy as np
import tensorflow as tf

## 분배 법칙이 성립되어 arr1의 각 요소에 temp 변수의 값이 더해짐
arr1 = np.array([1,2,3,4,5])
temp1 = 10
print(arr1 + temp1) # [11 12 13 14 15]


## 동일한 차원과 크기를 갖는 경우,
## 같은 위치에 해당하는 요소 사이에서 연산을 실행
arr2 = np.array([1,2,3,4,5])
arr3 = np.array([1,2,3,4,5])
print(arr2 + arr3)  # [2 4 6 8 10]


## 동일한 차원이지만 크기가 다른 경우,
## 크기가 작은 쪽의 모든 요소를 크기가 큰 요소에 분배하여 연산을 실행
## (분배가 성립할 수 있는 개수에서만 실행)
arr4 = np.array([1,2,3,4,5])
arr5 = np.array([5])
print(arr4 + arr5)  # [6 7 8 9 10]

## 텐서플로우 연산의 분배 법칙
## - 피연산자의 형태가 동일한 경우, 같은 위치에 해당하는 요소 사이에서 연산을 실행
## - 피연산자의 형태가 다른 경우, 한 텐서의 모든 요소에 연산을 실행 
## - 연산이 불가능한 형태인 경우, 에러 발생

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
    print(f"result = {result}") # result = [0 0 0 0 0 0 0 0 0 0]
    
    result = sess.run(X + list(range(1,11)))    
    print(f"result = {result}") # result = [1 2 3 4 5 6 7 8 9 10]
    
    result = sess.run(X_reshape + 7)
    print(f"result = {result}") # result = [[7 7] [7 7] [7 7] [7 7] [7 7]]

    result = sess.run(X_reshape + [2,5])
    print(f"result = {result}") # result = [[2 5] [2 5] [2 5] [2 5] [2 5]]


    ## 5 x 2 형태의 텐서에 크기가 3인 리스트는 더할 수 없음 
    # result = sess.run(X_reshape + [2,5,7])
    # print(f"result = {result}")
