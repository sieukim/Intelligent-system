# -*- coding: utf-8 -*-

## 강의 주제: 연산 텐서 with Input()

import tensorflow as tf

## 1. 실행 매개변수 텐서 정의
num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)


## 2. 연산 텐서 정의 
add = tf.add(num1, num2)
subtract = tf.subtract(num1, num2)
multiply = tf.multiply(num1, num2)
div = tf.divide(num1, num2)
mod = tf.mod(num1, num2)


## 3. 세션 생성 및 텐서 실행, 세션 종료
with tf.compat.v1.Session() as sess:
    n1 = float(input("첫 번째 숫자를 입력하세요 : "))
    n2 = float(input("두 번째 숫자를 입력하세요 : "))

    feed_dict = {num1: n1, num2: n2}

    result = sess.run(add, feed_dict=feed_dict)
    print(f'tf.add({n1}, {n2}) => {result}')
    
    result = sess.run(subtract, feed_dict=feed_dict)
    print(f'tf.subtract({n1}, {n2}) => {result}')    

    result = sess.run(multiply, feed_dict=feed_dict)
    print(f'tf.multiply({n1}, {n2}) => {result}')

    result = sess.run(div, feed_dict=feed_dict)
    print(f'tf.div({n1}, {n2}) => {result}')
    
    result = sess.run(mod, feed_dict=feed_dict)
    print(f'tf.mod({n1}, {n2}) => {result}')
    