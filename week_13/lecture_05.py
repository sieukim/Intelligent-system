# -*- coding: utf-8 -*-

### 강의 주제: 비지도 학습 - DBSCAN

##
## 1. 데이터 생성
##
## make_moons()
## - 초승달 모양 군집(클러스터) 두 개 형상을 갖는 데이터를 생성하는 함수 
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, # 표본 데이터의 수
                  noise=0.05,    # 잡음의 크기(0인 경우 반원을 이룸)
                  random_state=0)

print(X[:10])
print(y[:10])

## 2. 데이터 분포 확인 
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1])
plt.show()


## 3. 군집 분석 수행 - KMeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, 
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
km.fit(X)
y_cluster = km.predict(X)

## 군집 분포 확인 
plt.scatter(X[y_cluster==0, 0],
            X[y_cluster==0, 1],
            s=50, c='lightgreen',
            marker='s', label='Cluster 1')

plt.scatter(X[y_cluster==1, 0],
            X[y_cluster==1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=100, c='red',
            marker='*', label='Center')

plt.legend()
plt.grid()
plt.show()


## 4. 군집 분석 수행 - AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2)
y_cluster = ac.fit_predict(X)

## 군집 분포 확인 
plt.scatter(X[y_cluster==0, 0],
            X[y_cluster==0, 1],
            s=50, c='lightgreen',
            marker='s', label='Cluster 1')

plt.scatter(X[y_cluster==1, 0],
            X[y_cluster==1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.legend()
plt.grid()
plt.show()

## => make_moons()로 생성한 데이터의 경우, KMeans 또는
##    AgglomerativeClustering를 사용했을 때, 각각의
##    초승달 모양으로 군집이 나눠지지 않음
## => DBSCAN으로 해결 가능

##
## 5. 군집 분석 수행 - DBSCAN
##
## DBSCAN
## - 데이터 간의 밀도를 이용하여 군집의 형성 여부를
##   결정하는 알고리즘을 구현한 클래스 
## - n_clusters 가 자동으로 결정됨
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,
            min_samples=5,      # 한 군집 당 최소 데이터 개수
            metric='euclidean') # 판단 기준

## 군집 결과 생성 
y_cluster = db.fit_predict(X)

## 군집 분포 확인 
plt.scatter(X[y_cluster==0, 0],
            X[y_cluster==0, 1],
            s=50, c='lightgreen',
            marker='s', label='Cluster 1')

plt.scatter(X[y_cluster==1, 0],
            X[y_cluster==1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.legend()
plt.grid()
plt.show()