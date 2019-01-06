import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

original_img=cv2.imread("img2.bmp")

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Display the original histogram
img_size = gray_img.shape
heights = img_size[0]
widths = img_size[1]

store = np.zeros((256,),dtype=np.int32)
saved = np.zeros((256,),dtype=np.int32)
for i in range(heights):
    for j in range(widths):
       k = gray_img[i,j]
       store[k] += 1
x = np.arange(0,256)
#plt.bar(x,store,color="gray",align="center")
#plt.show()

# Define the size of filter
size = 3
k_baru = 0
l = np.zeros(shape=(3,3))
hasil = np.zeros(shape=(3,3))
f = 0
g = 0
gauss = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
kuntul = 0
kantal = 0

l_baru = 0
m = 0

# Add the padding around the image
padding = np.around(size // 2)
gray_img = cv2.copyMakeBorder(gray_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

# Get the new image size
shape=gray_img.shape
height_new=shape[0]
width_new=shape[1]

# Define the matrix for store and saved the cumulative distribution function
store = np.zeros((256,),dtype=np.int32)
saved = np.zeros((256,),dtype=np.int32)

# Copy the original image, used to display the new result
gray_img_copy = gray_img.copy()

# Looping for the entire image
for a in range(1,height_new-1):
    for b in range(1, width_new-1):
        # Looping for the entire window size
        f = 0
        for c in range(a-1, a+2):
            g = 0
            for d in range(b-1, b+2):
                # Get the pixel value in the current pixel position
                l[f, g] = gray_img[c, d]
                g += 1
            f += 1

        #print("gauss")
        #print(gauss)
        
        #print("l")
        #print(l)
        
        hasil = gauss*l
        #print("hasil")
        #print(hasil)

        kuntul = hasil.sum()
        #print("kuntul")
        #print(kuntul)

        kantal = kuntul / 16
        #print("hasil2")
        #print(kantal)
        
        #print("Save")

        gray_img_copy[a, b] = kantal
        
        l = np.zeros(shape=(3,3))
        hasil = np.zeros(shape=(3,3))

cv2.imshow('Display the result', gray_img_copy)
cv2.imwrite("result_local.jpg", gray_img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()        


