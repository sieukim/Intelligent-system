# -*- coding: utf-8 -*-

## 강의 주제: 예제 - 구구단 출력 

import tensorflow as tf

## 구구단의 단 또는 곱해지는 수를 증가시키기 위한 상수 텐서 정의 
nStep = tf.constant(1)

## 구구단의 단을 의미하는 변수 텐서 정의
x = tf.Variable(2)

## 곱해지는 수를 의미하는 변수 텐서 정의
y = tf.Variable(1)

## 현재 구구단의 단을 nStep만큼 증가한 값을 반환하는 연산 텐서 정의
addX = tf.add(x, nStep)

## 현재 곱해지는 수를 nStep만큼 증가한 값을 반환하는 연산 텐서 정의
addY = tf.add(y, nStep)

## 현재 구구단의 단을 addX()를 실행하여 나온 값으로 설정하는 연산 텐서 정의
updateX = tf.assign(x, addX)

## 현재 곱해지는 수를 addY()를 실행하여 나온 값으로 설정하는 연산 텐서 정의
updateY = tf.assign(y, addY)

## 구구단의 결과를 반환하는 연산 텐서 정의
result = tf.multiply(x, y)

## 곱해지는 수를 1로 초기화하는 연산 텐서 정의
initY = tf.assign(y, 1)

## 모든 변수 텐서를 초기화하기 위한 연산 텐서 정의
init_variables = tf.global_variables_initializer()

## 세션 생성 및 텐서 실행, 세션 종료
with tf.compat.v1.Session() as sess:
    ## 모든 변수 텐서 초기화
    sess.run(init_variables)
    
    for i in range(2, 10):
        ## 현재 구구단의 단과 곱해지는 수(=1)
        x_current, y_current = sess.run(x), sess.run(initY)
        
        print(f"{x_current}단을 출력합니다.")
        
        for j in range(9):
            ## 구구단 결과 출력
            print(f"{x_current} * {y_current} = {sess.run(result)}")
            ## 곱해지는 수 증가
            sess.run(updateY)
        
        ## 구구단의 단 증가
        sess.run(updateX)