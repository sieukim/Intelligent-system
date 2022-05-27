# -*- coding: utf-8 -*-

### 강의 주제: 비지도 학습 - KMeans

### 비지도 학습
### - 지도 학습과 달리 종속 변수(정답 데이터)를 제공하지 않고
###   데이터에 대한 학습을 수행하는 기법
### - 학습 결과는 주관적인 판단으로 처리
###
### 비지도 학습 종류
### - 차원 축소
###   데이터에 포함된 특성 중 유의미한 값을 추출.
###   데이터에 포함된 특성을 대표하는 값을 반환
###   주로 시각화, 소셜 네트워크 분석, 이미지 분석, RGB 등에 사용함
### - 군집 분석
###   데이터(샘플)의 유사성을 비교하여 동일한 특성으로 구성된
###   데이터(샘플)들을 하나의 군집으로 처리하는 기법
###
### 비지도 학습 유의 사항
### - 학습 결과를 100% 신뢰할 수 없음
### - 매번 다른 결과가 나올 수 있음 

### 군집 분석을 활용하여 데이터의 클러스터링 처리 과정을 확인
### - 지도 학습에서 군집 분석의 결과를 활용하는 방법 

##
## 1. 데이터 생성
##
## make_blobs()
## - 등방성 가우시안 정규분포를 이용해 가상 데이터를 생성하는 함수
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,    # 표본 데이터의 수
                  n_features=2,     # 독립 변수의 수
                  centers=3,        # 생성할 군집(클러스터)의 수
                  cluster_std=0.5,  # 군집(클러스터)의 표준 편차 
                  shuffle=True,     # 데이터 셔플
                  random_state=0)

print(X[:10])
print(y[:10])


## 2. 데이터 분포 확인 
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], 
            X[:, 1], 
            c='white',
            marker='o', 
            edgecolor='black',
            s=50)
plt.grid()
plt.show()

##
## 3. 군집 분석 수행
##
## KMeans(최근접 이웃 알고리즘)
## - 가장 많이 사용하는 군집 분석 클래스
## - 알고리즘이 단순하고 변경에 용이함(수정 사항의 반영이 손 쉬움)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,   # 군집(클러스터)의 수
            init='random',  # 초기 중심점 설정
            n_init=10,      # 초기 중심점 선택 반복 횟수
            max_iter=300,   # 학습 최대 반복 횟수 
            random_state=0)

## KMeans의 fit 메소드의 동작 방식
## - n_clusters에 정의된 개수만큼 포인트를 지정하여
##   최적의 위치를 찾는 검색 과정을 수행 
km.fit(X)

## 군집의 결과를 생성
y_cluster = km.predict(X)
print(y_cluster)

## 군집 분포 확인 
plt.scatter(X[y_cluster==0, 0],
            X[y_cluster==0, 1],
            s=50, c='lightgreen',
            marker='s', label='Cluster 1')

plt.scatter(X[y_cluster==1, 0],
            X[y_cluster==1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.scatter(X[y_cluster==2, 0],
            X[y_cluster==2, 1],
            s=50, c='lightblue',
            marker='v', label='Cluster 3')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=100, c='red',
            marker='*', label='Center')

plt.legend()
plt.grid()
plt.show()

