import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from skimage import io

img = io.imread('Data/118035a.jpg').astype(np.float32)
img = img / 255
rows, cols, _ = img.shape

img0 = img.reshape(rows * cols, 3)
sigma = np.float32(0.1)
shape1 = []
shape2 = []
elem = []
for i in range(rows):
  for j in range(cols):
    idx = i * cols + j
    for k, l in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      if 0 <= (i + k) < rows and 0 <= (j + l) < cols:
        nbr = (i + k) * cols + (j + l)
        num = np.sum((img0[idx] - img0[nbr])**2)
        a = np.exp(-num / (sigma**2))
        shape1.append(idx)
        shape2.append(nbr)
        elem.append(a)
A = sparse.csr_matrix((np.array(elem, dtype = np.float32), (shape1, shape2)), shape = (rows * cols, rows * cols))

## 1-(a)
def custom_clustering(A, nclusters):
  d = np.array(A.sum(axis = 1), dtype = np.float32).flatten()
  Dinvhalf = sparse.diags(1 / np.sqrt(d + 1e-10))
  L = Dinvhalf @ A @ Dinvhalf
  U, S, _ = svds(sparse.eye(A.shape[0], dtype = np.float32) - L, k = nclusters)
  U = U[:, np.argsort(-S)]
  ev_norm = U / np.sqrt(np.sum(U ** 2, axis = 1))[:, np.newaxis]
  km = KMeans(n_clusters = nclusters)
  labels = km.fit_predict(ev_norm)
  return labels

labels = custom_clustering(A, 10)
labels0 = labels.reshape(rows, cols)
plt.imshow(labels0)
plt.colorbar(label = 'Cluster')
plt.show()

## 1-(b)
def custom_clustering_mean(img, labels, rows, cols, nclusters):
  img0 = img.reshape(-1, 3)
  means = np.zeros_like(img0)
  for i in range(nclusters):
    means[labels == i] = np.mean(img0[labels == i], axis = 0)

  means0 = means.reshape(rows, cols, 3)
  return means0

means0 = custom_clustering_mean(img, labels, rows, cols, 10)
plt.imshow(means0)
plt.show()

## 1-(c)
labels = custom_clustering(A, 20)
labels0 = labels.reshape(rows, cols)
plt.imshow(labels0)
plt.colorbar(label = 'Cluster')
plt.show()

means0 = custom_clustering_mean(img, labels, rows, cols, 20)
plt.imshow(means0)
plt.show()