# -*- coding: utf-8 -*-

### 강의 주제: 비지도 학습 - 병합 군집 

## 병합 군집
## - 다수개의 소규모 군집을 생성하고 취합하여 하나로 병합
## - 인접한 위치의 군집 사이에서 발생함
## - 원하는 개수의 군집으로 최정 처리를 완료함 

## 1. 데이터 생성
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
## 3. 병합 군집 분석 수행 
##
## AgglomerativeClustering
## - 병합 군집을 처리하는 클래스 
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)

## AgglomerativeClustering의 fit_predict 메소드의 동작 방식
## - n_clusters에 정의된 개수만큼 소규모의 군집들을 계속해서 병합한 
##   후, 정의된 개수에 도달하면 해당 정보를 반환함
y_cluster = ac.fit_predict(X)
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

plt.legend()
plt.grid()
plt.show()

