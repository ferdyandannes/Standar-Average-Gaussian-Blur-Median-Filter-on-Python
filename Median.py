import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

original_img=cv2.imread("img3.png")

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

final = gray_img[:]

for y in range(len(gray_img)):
    for x in range(y):
        final[y,x]=gray_img[y,x]

members=[gray_img[0,0]]*9
for y in range(1,gray_img.shape[0]-1):
    for x in range(1,gray_img.shape[1]-1):
        members[0] = gray_img[y-1,x-1]
        members[1] = gray_img[y,x-1]
        members[2] = gray_img[y+1,x-1]
        members[3] = gray_img[y-1,x]
        members[4] = gray_img[y,x]
        members[5] = gray_img[y+1,x]
        members[6] = gray_img[y-1,x+1]
        members[7] = gray_img[y,x+1]
        members[8] = gray_img[y+1,x+1]

        members.sort()
        final[y,x]=members[4]

cv2.imshow('Display the result', gray_img)
cv2.imwrite("result_local.jpg", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
