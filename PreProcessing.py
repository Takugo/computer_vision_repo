import cv2 as cv
import os
import numpy as np

# Utility: convert image to binary
def convert_to_binary(path=None, img=None, kernel_value=5, thresholdBinary = 150):
    if path != None:
        templ_filePath = path
        templ = cv.imread(templ_filePath, 0)
    else:
        templ = img

    kernel = (kernel_value, kernel_value)  # default Gaussian Kernel is (5, 5)
    templ_gaussianBlur = cv.GaussianBlur(templ, kernel, sigmaX=1, sigmaY=1)
    if len(templ_gaussianBlur.shape) == 3:
        templ = cv.cvtColor(templ_gaussianBlur, cv.COLOR_BGR2GRAY)
    else:
        templ = templ_gaussianBlur
    ret_templ, templ_binary = cv.threshold(templ, 0.0, 255.0, cv.THRESH_BINARY | cv.THRESH_OTSU)
    if thresholdBinary != 150:
        ret_templ, templ_binary = cv.threshold(templ, thresholdBinary, 255, cv.THRESH_BINARY)

    return templ_binary, templ_gaussianBlur

# convert image into matrix
def image_to_matrix(img):
    img_mat = np.matrix(img)
    return img_mat
