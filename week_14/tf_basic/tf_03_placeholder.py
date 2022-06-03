# -*- coding: utf-8 -*-

## 강의 주제: 텐서 타입 - 실행 매개변수 

import tensorflow as tf

## 1. 실행 매개변수 텐서 정의 
## - tf.placeholder를 사용하여 정의
## - 세션 메소드 run()이 실행될 때, 값이 전달되어야 하는 텐서
## - 각각의 run()을 실행할 때, 다른 값을 사용하여 실행할 수 있음
## - 일반적으로 학습 데이터, 테스트 데이터를 전달하기 위해 사용
## - 반드시 run()의 매개변수로 feed_dict를 전달해야 함
##   (값이 필요하지 않은 경우엔 생략 가능)
## - 프로그램 실행 중 값의 변경이 허용 안 됨
ph_1 = tf.placeholder(tf.float32)

ph_2 = tf.placeholder(tf.float32, shape=[None])

ph_3 = tf.placeholder(tf.float32, shape=[None, 2])


## 2. 연산텐서 정의 

## tf.square(tensor)
## - 매개변수로 전달 받은 텐서의 제곱 값을 반환하는 연산 텐서를 반환 
squar_tensor_1 = tf.square(ph_1)

squar_tensor_2 = tf.square(ph_2)

## tf.add(tensor)
## - 매개변수로 전달 받은 텐서들의 합계를 반환하는 연산 텐서를 반환 
plus_tensor = tf.add(squar_tensor_1, squar_tensor_2)


## 3. 세션 생성 및 텐서 실행, 세션 종료
## - 자동 종료를 위해 with 절을 활용
## - run() 메소드 실행시 feed_dict를 매개변수로 넣어주어야 하며,
##   모든 실행 매개변수 텐서에 대한 값을 전달해야 함 
with tf.compat.v1.Session() as sess:
    result = sess.run(ph_1, feed_dict={ph_1 : 17})
    print(f"result = {result}") # result = 17.0
    
    result = sess.run(ph_2, feed_dict={ph_2 : [1,2,3,5,6,7,8,9,10,11,12]})
    print(f"result = {result}") # result = [1. 2. 3. 5. 6. 7. 8. 9. 10. 11. 12.]
    
    result = sess.run(ph_3, feed_dict={ph_3 : [[1,2],[3,4],[5,6]]})
    print(f"result = {result}") # result = [[1. 2.] [3. 4.] [5. 6.]]
    
    result = sess.run(plus_tensor, feed_dict={ph_1 : 17, ph_2 : [10,20]})
    print(f"result = {result}") # result = [389. 689.]
