import matplotlib.pyplot as plt 
import numpy as np 
import cv2



original_image = cv2.imread('petri.jpg')

img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1, 3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

k = 4 
attempts = 100

ret, label, center = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)


# with np.printoptions(threshold=np.inf):
#     with open('label.txt', 'w') as f:
#         f.write(str(label))

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Imagem segmentada com  K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()

