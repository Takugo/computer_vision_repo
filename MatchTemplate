import cv2 as cv
import numpy as np
from skimage.measure import compare_ssim

# Using SSIM to compare two images, returns a metric
def cal_ssim(source, target):  # calculate the SSIM
    target = cv.resize(target, (source.shape[1], source.shape[0]))  # shape[0] = height, shape[1] = width. resize(img, Size(width, height))
    target, _, _ = cv.split(target) if len(target.shape) == 3 else target
    if len(source.shape) == 3:
        source, _, _ = cv.split(source)
    else:
        source = source
    ssim_result = compare_ssim(source, target, multichannel=True)
    return ssim_result
