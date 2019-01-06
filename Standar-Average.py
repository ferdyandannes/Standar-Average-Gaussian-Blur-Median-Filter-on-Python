import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

original_img=cv2.imread("img1.tif")

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
plt.bar(x,store,color="gray",align="center")
plt.show()

# Define the size of filter
size = 5
k_baru = 0

if size == 3:
    divider = 9
elif size == 5:
    divider = 25
elif size == 7:
    divider = 49;

# Add the padding around the image
padding = np.around(size // 2)
gray_img = cv2.copyMakeBorder(gray_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

# Get the new image size
shape=gray_img.shape
height_new=shape[0]
width_new=shape[1]

print(height_new)
print(width_new)

# Define the matrix for store and saved the cumulative distribution function
store = np.zeros((256,),dtype=np.int32)
saved = np.zeros((256,),dtype=np.int32)

# Copy the original image, used to display the new result
gray_img_copy = gray_img.copy()

# Looping for the entire image
for a in range(2,height_new-3):
    for b in range(2, width_new-3):
        # Looping for the entire window size
        for c in range(a-2, a+3):
            for d in range(b-2, b+3):
                # Get the pixel value in the current pixel position
                k = gray_img[c, d]
                k_baru += k
                
        k_baru = k_baru / divider;
        gray_img_copy[a, b] = k_baru

        # Reset the stored value
        sum_hist = np.zeros((256,),dtype=np.int32)
        saved = np.zeros((256,),dtype=np.int32)
        k_baru = 0
        store = np.zeros((256,),dtype=np.int32)

# Show and save the result
# Display the original histogram
store = np.zeros((256,),dtype=np.int32)
saved = np.zeros((256,),dtype=np.int32)
for i in range(height_new):
    for j in range(width_new):
       k = gray_img_copy[i,j]
       store[k] += 1
x = np.arange(0,256)
plt.bar(x,store,color="gray",align="center")
plt.show()

cv2.imshow('Display the result', gray_img_copy)
cv2.imwrite("result_local.jpg", gray_img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


FILTER 7
import cv2
import os
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
plt.bar(x,store,color="gray",align="center")
plt.show()

# Define the size of filter
size = 7
k_baru = 0

if size == 3:
    divider = 9
elif size == 5:
    divider = 25
elif size == 7:
    divider = 49;

# Add the padding around the image
padding = np.around(size // 2)
gray_img = cv2.copyMakeBorder(gray_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

# Get the new image size
shape=gray_img.shape
height_new=shape[0]
width_new=shape[1]

print(height_new)
print(width_new)

# Define the matrix for store and saved the cumulative distribution function
store = np.zeros((256,),dtype=np.int32)
saved = np.zeros((256,),dtype=np.int32)

# Copy the original image, used to display the new result
gray_img_copy = gray_img.copy()

# Looping for the entire image
for a in range(3,height_new-2):
    for b in range(3, width_new-2):
        # Looping for the entire window size
        for c in range(a-3, a+3):
            for d in range(b-3, b+3):
                # Get the pixel value in the current pixel position
                k = gray_img[c, d]
                k_baru += k
                
        k_baru = k_baru / divider;
        gray_img_copy[a, b] = k_baru

        # Reset the stored value
        sum_hist = np.zeros((256,),dtype=np.int32)
        saved = np.zeros((256,),dtype=np.int32)
        k_baru = 0
        store = np.zeros((256,),dtype=np.int32)

# Show and save the result
# Display the original histogram
store = np.zeros((256,),dtype=np.int32)
saved = np.zeros((256,),dtype=np.int32)
for i in range(height_new):
    for j in range(width_new):
       k = gray_img_copy[i,j]
       store[k] += 1
x = np.arange(0,256)
plt.bar(x,store,color="gray",align="center")
plt.show()

cv2.imshow('Display the result', gray_img_copy)
cv2.imwrite("result_local.jpg", gray_img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
