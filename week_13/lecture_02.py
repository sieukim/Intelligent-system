# -*- coding: utf-8 -*-

### 강의 주제: 비지도 학습 - KMeans + 최적의 군집(클러스터) 개수 검색 

## 최적의 군집(클러스터) 개수 검색 방법
## - 엘로우 방법을 활용하여 검색 
## - 아래와 같이 반복문을 돌며 inertia_ 값을 저장하고,
##   분포를 살펴본 후 급격히 낮아지는 포인트를 탐색 

## 1. 데이터 생성
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,    # 표본 데이터의 수
                  n_features=2,     # 독립 변수의 수
                  centers=5,        # 생성할 군집(클러스터)의 수
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

## 3. 군집 분석 수행 + 최적의 군집(클러스터) 개수 검색
from sklearn.cluster import KMeans

## inertia_ 속성 리스트
values = []

## 군집(클러스터)개수를 1에서 10까지 
## 늘려가며 inertia_ 값을 저장
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    
    ## inertia_ 저장 
    ## - 군집(클러스터) 내 각 클래스의 SSE 값 
    values.append(km.inertia_)
    
print(values)

## inertia_ 분포 확인 
plt.plot(range(1, 11), values, marker='o')
plt.xlabel('numbers of cluster')
plt.ylabel('inertia_')
plt.show()
