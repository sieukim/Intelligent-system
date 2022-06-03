# -*- coding: utf-8 -*-

## 강의 주제: 두 행렬의 곱 

import tensorflow as tf

## tf.matmul(matrix1, matrix2)
## - 입력받은 두 텐서가 행렬을 저장할 때,
##   행렬곱의 결과를 반환하는 함수 

## 1. 변수 텐서 정의 
X1 = tf.Variable([[1,2],[3,4]])         # 2 X 2
X2 = tf.Variable([[5,6,7,8]])           # 1 X 4


## 2. 크기 변경 
X2_reshape_1 = tf.reshape(X2, [2, -1])  # 2 X 2


## 3. 행렬 곱 연산 텐서 정의
matmul_1 = tf.matmul(X1, X2_reshape_1)  # (2 x 2) * (2 x 2)

X2_reshape_2 = tf.reshape(X2, [4, -1])  # 4 X 1

## 행렬 곱의 규칙에 위배되기 때문에 에러가 발생됨
#matmul_2 = tf.matmul(X1, X2_reshape_2) # (2 x 2) * (4 x 1)


## 4. 변수 텐서 초기화 연산 텐서 정의
init_variables = tf.global_variables_initializer()


## 5. 세션 생성 및 텐서 실행, 세션 종료
with tf.compat.v1.Session() as sess :
    ## 변수 텐서 초기화 
    sess.run(init_variables)
    
    result = sess.run(matmul_1)
    print(f"result = {result}") # result = [[19 22] [43 50]]
    