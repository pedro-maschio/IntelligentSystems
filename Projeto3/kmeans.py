import matplotlib.pyplot as plt 
import numpy as np 
import cv2



imagem_original = cv2.imread('petri.jpg')

img = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)

imagem_vectorizada = img.reshape((-1, 3))
imagem_vectorizada = np.float32(imagem_vectorizada)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

k = 4 
tentativas = 100

ret, label, center = cv2.kmeans(imagem_vectorizada, k, None, criteria, tentativas, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]

with np.printoptions(threshold=np.inf):
    with open('kmeansvectorized.txt', 'w') as f:
        f.write(str(imagem_vectorizada))

print(img.shape)
result_image = res.reshape((img.shape))

with np.printoptions(threshold=np.inf):
    with open('kmeansresult_image.txt', 'w') as f:
        f.write(str(result_image))

tamanho_da_figura = 20
plt.figure(figsize=(tamanho_da_figura,tamanho_da_figura))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Imagem segmentada com  K = %i' % k), plt.xticks([]), plt.yticks([])
#plt.show()
