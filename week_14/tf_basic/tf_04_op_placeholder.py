# -*- coding: utf-8 -*-

## 강의 주제: 연산 텐서 with 실행 매개변수 텐서 

import tensorflow as tf

## 1. 실행 매개변수 텐서 정의
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


## 2. 연산 텐서 정의 
## - 아래 연산텐서를 실행하기 위해서는
##   실행 매개변수 텐서 x, y의 값을 반드시 전달해야 함 
add = tf.add(x,y)               # x+y
subtract = tf.subtract(x,y)     # x-y
multiply = tf.multiply(x, y)    # x*y
div = tf.divide(x,y)            # x/y
mod = tf.mod(x,y)               # x%y


## 3. 세션 생성 및 텐서 실행, 세션 종료
## - feed_dict는 실행 매개변수 텐서의 변수명과 대소문자까지 일치한 속성을 사용해야 함 
with tf.compat.v1.Session() as sess:
    feed_dict = {x:10, y:5}
    result = sess.run(add, feed_dict=feed_dict)
    print(f"tf.add(10, 5) => {result}")         # tf.add(10, 5) => 15.0
    
    feed_dict = {x:100, y:50}
    result = sess.run(subtract, feed_dict=feed_dict)
    print(f"tf.subtract(100, 50) => {result}")  # tf.subtract(100, 50) => 50.0

    feed_dict = {x:7, y:3}
    result = sess.run(multiply, feed_dict=feed_dict)
    print(f"tf.multiply(7, 3) => {result}")     # tf.multiply(7, 3) => 21.0
    
    feed_dict = {x:10, y:3}
    result = sess.run(div, feed_dict=feed_dict)
    print(f"tf.div(10, 3) => {result}")         # tf.div(10, 3) => 3.333...

    feed_dict = {x:10, y:5}
    result = sess.run(mod, feed_dict=feed_dict)
    print(f"tf.mod(10, 5) => {result}")         # tf.mod(10, 5) => 0.0  
