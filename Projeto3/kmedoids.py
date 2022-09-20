from sklearn_extra.cluster import KMedoids
import cv2
import numpy as np 

original_image = cv2.imread('petri.jpg')
scale = 30 # scale in percentage

img = cv2.resize(original_image, (int(original_image.shape[1]*scale/100), int(original_image.shape[0]*scale/100)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1, 3))
vectorized = np.float32(vectorized)

print(vectorized.shape)

with np.printoptions(threshold=np.inf):
    with open('imagekmedoids.txt', 'w') as f:
        f.write(str(vectorized))

kmedoids = KMedoids(n_clusters=4, random_state=0).fit(vectorized)

with np.printoptions(threshold=np.inf):
    with open('imagekmedoidslabels.txt', 'w') as f:
        f.write(str(kmedoids.labels_))
        f.write(str(kmedoids.cluster_centers_))
