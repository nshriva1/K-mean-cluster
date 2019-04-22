from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets


iris = datasets.load_iris()
X = pd.DataFrame(data = iris.data)


iris_features = X.iloc[:, [0,1, 2, 3]].values

no_of_clust = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 200, n_init = 10, random_state = 0)
    kmeans.fit(iris_features)
    no_of_clust.append(kmeans.inertia_)
 
    
plt.plot(range(1, 11), no_of_clust,color="black")
plt.xlabel('Number of Clusters')
plt.ylabel('Values within Cluster Sum of Squares')
plt.title('The Elbow Method')
plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 200, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(iris_features)

label_color = ["Yellow","Cyan","Navy"]
label = ['Iris-setosa', 'Iris-versicolour','Iris-virginica']


for plot_no in range(3):
    plt.scatter(iris_features[y_kmeans == plot_no, 0], iris_features[y_kmeans == plot_no, 1], s = 100, c = label_color[plot_no], label = label[plot_no])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()
plt.show()
