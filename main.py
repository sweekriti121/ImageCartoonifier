#Importing Modules
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#Load Image
img=cv.imread('Image1.jpeg')
ReSized1 = cv.resize(img, (300, 300))
cv.imshow("Original Image",ReSized1)
#Grayscale
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ReSized2=cv.resize(gray, (300, 300))
cv.imshow("Grayscaled Image",ReSized2)
#Smoothening
median=cv.medianBlur(gray,1)
ReSized3=cv.resize(median, (300, 300))
cv.imshow("Smoothened Image",ReSized3)
#Edges
getEdge = cv.adaptiveThreshold(median, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11 ,11)
ReSized4=cv.resize(getEdge, (300, 300))
cv.imshow("Edged Image",ReSized4)
#Bilateral filter
bilateral=cv.bilateralFilter(img,9,150,150)
ReSized5=cv.resize(bilateral, (300, 300))
cv.imshow("Bilateral Image",ReSized5)
#Masking
cartoonifiedImage=cv.bitwise_and(bilateral,bilateral,mask=getEdge)
ReSized6=cv.resize(cartoonifiedImage, (300, 300))
cv.imshow("Cartoonified Image",ReSized6)
cv.waitKey(0)