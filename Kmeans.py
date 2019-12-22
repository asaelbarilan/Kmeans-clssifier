
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Open the file
xlsx = pd.ExcelFile('./Bloodtests.xlsx')

# Get the first sheet as an object
sheet1 = xlsx.parse(0)

# Set the Key column as the index
sheet1.set_index("Key", inplace=True)
variables1 = ["K","Hgb","WBC"]
EMR3d1 = sheet1.loc[:, variables1].values


class Kmeans:
  def __init__(self,k):
      self.K_clusters = k

  def fit(self,X,iterations=30,treshold=0.05):
    # your code goes here
    self.train_samples=X
    self.iter=iterations
    self.treshold=treshold

  def predict(self, X):
    # your code goes here
    k=self.K_clusters
    Xdf=pd.DataFrame(X)
    centers_old = Xdf.sample(n = self.K_clusters,axis=0,random_state=2).values
    centers_new = np.zeros_like(centers_old)
    distance = np.zeros((X.shape[0], k), dtype='float32')
    y_pred = np.zeros((X.shape[0]), dtype='float32')
    error=self.treshold+1
    for j in range(self.iter):
        if error>self.treshold:
            for i in range(k):
                distance[:, i] = np.linalg.norm(X -centers_old[i,:], axis=1)
            y_pred = np.argmin(distance, axis=1)
            for k_cluster in range(k):
                centers_new[k_cluster]=X[y_pred==k_cluster].mean(axis=0)
            error = abs(np.linalg.norm(centers_new - centers_old))
            centers_old=centers_new
            self.cluster_centers=centers_old
        else:
            break
    return y_pred
# kobject=Kmeans(3)
# kobject.fit(EMR3d1,300)
# y_predicted=kobject.predict(EMR3d1)

def make_chart (chart_data, y_pred, labels):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.scatter(chart_data[:,0], chart_data[:,1], chart_data[:,2], c=y_pred)

#make_chart(EMR3d1, y_predicted, variables1)



from sklearn import metrics
from scipy.spatial.distance import cdist
X=EMR3d1

k_list=list(range(2, 21))
calinski_score=np.zeros((len(k_list),1))
silhouette_score=np.zeros((len(k_list),1))
distortions=np.zeros((len(k_list),1))
#metrics.calinski_harabaz_score(X, labels)
for i,k in enumerate(k_list):
    kobject = Kmeans(k)
    kobject.fit(EMR3d1, 30)
    y_predicted = kobject.predict(EMR3d1)
    calinski_score[i]=metrics.calinski_harabaz_score(X, y_predicted)
    silhouette_score[i]=metrics.silhouette_score(X, y_predicted)
    distortions[i]=(sum(np.min(cdist(X, kobject.cluster_centers, 'euclidean'), axis=1)) / X.shape[0])
    print( k, metrics.calinski_harabaz_score(X, y_predicted),metrics.silhouette_score(X, y_predicted))

print(k_list[calinski_score.argmax()],'calinski_score')
print(k_list[silhouette_score.argmax()],'silhouette_score')
plt.plot(k_list,distortions)
plt.xlabel('k values')
plt.ylabel('distortion')
plt.show()

