# -*- coding: utf-8 -*-

## 강의 주제: 연산 텐서 with 변수 텐서 

import tensorflow as tf

## 1. 변수 텐서 정의
x = tf.Variable(10)
y = tf.Variable(7)

## 2. 연산 텐서 정의 
add = tf.add(x, y)
subtract = tf.subtract(x, y)
multiply = tf.multiply(x, y)
div = tf.divide(x, y)
mod = tf.mod(x, y)

## 3. 세션 생성 및 텐서 실행, 세션 종료
## - tf.Variable을 통해 생성된 텐서는 
##   tf.global_variable_initializer()를
##   통한 초기화 수행 필요 
with tf.compat.v1.Session() as sess:
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    
    x, y = sess.run(x), sess.run(y)
    
    result = sess.run(add)
    print(f'tf.add({x}, {y}) => {result}')      # tf.add(10, 7) => 17
    
    result = sess.run(subtract)
    print(f'tf.subtract({x}, {y}) => {result}') # tf.subtract(10, 7) => 3
    
    result = sess.run(multiply)
    print(f'tf.multiply({x}, {y}) => {result}') # tf.multiply(10, 7) => 70
    
    result = sess.run(div)
    print(f'tf.div({x}, {y}) => {result}')      # tf.div(10, 7) => 1.428...
    
    result = sess.run(mod)
    print(f'tf.mod({x}, {y}) => {result}')      # tf.mod(10, 7) => 3
    
