# -*- coding: utf-8 -*-

## 강의 주제: 텐서 타입 - 상수
 
import tensorflow as tf

## 1. 상수 텐서 정의 
## - tf.constant를 사용하여 정의
## - 프로그램 실행 중 값의 변경이 허용 안 됨
## - 선언시 반드시 초기화가 수행되어야 함
cons_1 = tf.constant(1)

cons_2 = tf.constant([2.0])

cons_3 = tf.constant([1,2,3])

cons_4 = tf.constant([1,2,3,4], shape=[2,2], dtype=tf.float64)


## 2. 세션 객체 생성 
sess = tf.compat.v1.Session()


## 3. 텐서 실행
result = sess.run(cons_1)
print(f"result = {result}")     # result = 1 

result = sess.run(cons_2)
print(f"result = {result}")     # result = [2.]

result = sess.run(cons_3)
print(f"result = {result}")     # result = [1 2 3]

result = sess.run(cons_4)
print(f"result = {result}")     # result = [[1. 2.] [3. 4.]]
print(f"result = {result[0]}")  # result = [1. 2.]
print(f"result = {result[1]}")  # result = [3. 4.]


## 4. 세션 종료
sess.close()
