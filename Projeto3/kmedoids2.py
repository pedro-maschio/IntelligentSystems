from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import cv2
import numpy as np 

original_image = cv2.imread('petri.jpg')

scale = 30 # scale in percentage

img = cv2.resize(original_image, (int(original_image.shape[1]*scale/100), int(original_image.shape[0]*scale/100)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((-1, 3))
img = np.float32(img)


# r, g, b = cv2.split(img)

# r = r/255
# g = g/255
# b = b/255

# print(r.shape)
# pca_r = PCA(n_components=50).fit_transform(r)
# print(pca_r.shape)

# pca_g = PCA(n_components=50).fit_transform(g)

# pca_b = PCA(n_components=50).fit_transform(b)



print(img.shape)
reduced_image = PCA(n_components=3).fit_transform(img)
print(reduced_image.shape)
kmedoids = KMedoids(n_clusters=4, random_state=0).fit(reduced_image)

print(kmedoids.cluster_centers_)