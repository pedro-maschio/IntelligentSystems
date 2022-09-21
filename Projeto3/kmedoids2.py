from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt 
import cv2
import numpy as np 

original_image = cv2.imread('petri.jpg')

scale = 20 # scala em porcentagem

img = cv2.resize(original_image, (int(original_image.shape[1]*scale/100), int(original_image.shape[0]*scale/100)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imagem_vectorizada = img.reshape((-1, 3))
imagem_vectorizada = np.float32(imagem_vectorizada)

# with np.printoptions(threshold=np.inf):
#     with open('kmedoidsimagem_vectorizada.txt', 'w') as f:
#         f.write(str(imagem_vectorizada))

k =4 
kmedoids = KMedoids(n_clusters=k, method='pam', random_state=0).fit(imagem_vectorizada)

centers = kmedoids.cluster_centers_

# with np.printoptions(threshold=np.inf):
#     with open('kmedoidslabels.txt', 'w') as f:
#         f.write(str(kmedoids.labels_))

res = centers[kmedoids.labels_.flatten()]

# with np.printoptions(threshold=np.inf):
#     with open('kmedoidsres.txt', 'w') as f:
#         f.write(str(res))

result_image = res.reshape((img.shape))

# with np.printoptions(threshold=np.inf):
#     with open('kmedoidsresult_image.txt', 'w') as f:
#         f.write(str(result_image))

tamanho_da_figura = 20
plt.figure(figsize=(tamanho_da_figura,tamanho_da_figura))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow((result_image * 255).astype(np.uint8))
plt.title('Imagem segmentada com  K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()