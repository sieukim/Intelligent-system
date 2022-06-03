# -*- coding: utf-8 -*-

## 강의 주제: 텐서 타입 - 변수 

import tensorflow as tf

## 1. 변수 텐서 정의 
## - tf.Variable을 사용하여 정의
## - 프로그램 실행 중 값이 변경될 수 있음
## - 기울기(가중치), 편향(절편)의 값을 저장하는
##   텐서 선언시 사용됨 
## - 반드시 세션을 통해 초기화를 수행해야 함 
var_1 = tf.Variable(1)

var_2 = tf.Variable([1, 2, 3])

## tf.fill(shape, value)
## - shape와 value를 입력받아 shape 형태의 텐서를 생성하고,
##   value 값으로 초기화하는 함수 
var_3 = tf.Variable(tf.fill([3,3], 3))

var_4 = tf.Variable([1, 2, 3, 4], dtype=tf.float64)


## 2. 연산 텐서 정의
## - var_1 텐서의 값을 11로 수정하는 연산 텐서
## - 세션 메소드 run()을 통해 연산을 수행할 수 있음 
## - ex) sess.run(var_1_assign)
var_1_assign = var_1.assign(11)


## 3. 세션 객체 생성 
sess = tf.compat.v1.Session()


## 4. 변수 텐서 초기화
## - tf.Variable을 통해 선언된 텐서가 하나라도 존재하는 경우
##   반드시 초기화를 수행해야 함 
init_variables = tf.global_variables_initializer()
sess.run(init_variables)


## 5. 텐서 실행 
result = sess.run(var_1)
print(f"result = {result}")     # result = 1

sess.run(var_1_assign)
result = sess.run(var_1)
print(f"result = {result}")     # result = 11

result = sess.run(var_2)
print(f"result = {result}")     # result = [1 2 3]

result = sess.run(var_3)
print(f"result = {result}")     # result = [[3 3 3] [3 3 3] [3 3 3]]

result = sess.run(var_4)
print(f"result = {result}")     # result = [1. 2. 3. 4.]
print(f"result = {result[1]}")  # result = 2.0


## 6. 세션 종료  
sess.close()
