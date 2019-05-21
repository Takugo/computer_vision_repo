# -*- coding: utf-8 -*-
# @Time : 2019/4/8 13:53
# @Author : Zehao Zhang -- RDRI
# @FileName: BlurDetection.py
# @Software: PyCharm

#imports
import cv2 as cv
import numpy as np
import PreProcessing as pp
import MatchTemplate as mt
import ParameterStructure as ps
from skimage import filters
"""
Function iqa() refers to <No-Reference Image Quality Assessment using Blur and Noise>
Function cal_nrss() refers to NRSS using SSIM
Function cal_reblur() refers to <No-reference quality assessment for blur image combined with re-blur range
and singular value decomposition>
"""

def cal_variance_of_laplacian(img):
    return cv.Laplacian(img, cv.CV_64F).var()



'''NRSS'''
# calculate the NRSS
# -- no-reference, no threshold
def cal_nrss(img):

    Ir = cv.GaussianBlur(img, ksize=(7, 7), sigmaX=6, sigmaY=6)

    grad_img = cv.Sobel(img, cv.CV_32FC1, 1, 1)
    grad_Ir = cv.Sobel(Ir, cv.CV_32FC1, 1, 1)

    block_cols = grad_img.shape[1] * 2 // 9
    block_rows = grad_img.shape[0] * 2 // 9

    N = 64
    max_stddev = 0.0
    pos = 0

    for idx in range(0, N):
        left_x = (idx % 8) * (block_cols // 2)
        left_y = (idx // 8) * (block_rows // 2)
        right_x = left_x + block_cols
        right_y = left_y + block_rows

        if left_x < 0:
            left_x = 0
        if left_y < 0:
            left_y = 0
        if right_x >= grad_img.shape[1]:
            right_x = grad_img.shape[1] - 1
        if right_y >= grad_img.shape[0]:
            right_y = grad_img.shape[0] - 1

        roi = grad_img[left_y : right_y, left_x : right_x]

        temp = roi.copy()
        mean, stddev = cv.meanStdDev(temp)

        if stddev[0] > max_stddev:
            max_stddev = float(stddev[0])
            pos = idx
            best_grad_img = temp
            best_grad_Ir = grad_Ir[left_y:right_y, left_x:right_x].copy()

    result = 1 - mt.cal_ssim(best_grad_img, best_grad_Ir)

    return result

'''Brener gradient function'''
# the larger metric value,
#  the more blur
def cal_brenner(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    x, y = img.shape
    metric = 0
    for i in range(x-2):
        for j in range(y-2):
            metric += (img[i+2, j] - img[i, j])**2
    return metric
'''Tenengrad gradient function'''
# Tenengrad gradient function
# The smaller metric value, the more blur
def cal_tenengrad(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    temp = pp.image_to_matrix(img)
    temp_sobel = filters.sobel(temp)
    source = np.sum(temp_sobel ** 2)
    metric = np.sqrt(source)
    return metric

'''Vollath function'''
# performance is not good
def cal_vollath(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    img = pp.image_to_matrix(img)
    x, y = img.shape
    source = 0
    for i in range(x-1):
        for j in range(y):
            source += img[i, j] * img[i + 1, j]
    metric = source - x * y * np.mean(img)
    return metric

'''EAV function/ PAV function'''
def cal_eav(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)

    height, width = img.shape
    metric = 0.0
    for y in range(1, height - 1):
        ptr_prev = y - 1
        ptr_cur = y
        ptr_next = y + 1
        for x in range(1, width - 1):
            metric += (abs(img[ptr_prev][x - 1] - img[ptr_cur][x]) * 0.707 + abs(img[ptr_prev][x] - img[ptr_cur][x])
                       + abs(img[ptr_prev][x+1] - img[ptr_cur][x]) * 0.707 + abs(img[ptr_next][x] - img[ptr_cur][x]) * 0.707
                       + abs(img[ptr_next][x] - img[ptr_cur][x]) + abs(img[ptr_next][x + 1] - img[ptr_cur][x]) * 0.707
                       + abs(img[ptr_cur][x - 1] - img[ptr_cur][x]) +
                       abs(img[ptr_cur][x + 1] - img[ptr_cur][x]))

    return metric/(height * width)


'''SMD function'''
# The larger returned value, the better
def cal_smd(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    metric = 0.0
    x, y = img.shape
    img = pp.image_to_matrix(img)

    for i in range(x-1):
        for j in range(y-1):
            metric += abs(int(img[i+1, j]) - int(img[i, j])) + abs(int(img[i, j]) - int(img[i+1, j]))

    return (metric/100)


'''SMD2 function'''
# The larger returned value, the better
def cal_smd2(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    metric = 0.0
    x, y = img.shape
    img = pp.image_to_matrix(img)

    for i in range(x-1):
        for j in range(y-1):
            metric += abs(int(img[i+1, j]) - int(img[i, j])) * abs(int(img[i, j]) - int(img[i+1, j]))
    return metric/1000


'''Energy gradient function'''
# The larger returned value, the better
def cal_energy_gradient(img):
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    metric = 0.0
    x, y = img.shape
    img = pp.image_to_matrix(img)

    for i in range(x-1):
        for j in range(y-1):
            metric += (int(img[i+1, j]) - int(img[i, j]))**2 * (int(img[i, j]) - int(img[i+1, j]))**2

    return metric/100000

'''Reblur function'''
# Log-Gabor + Blur estimation function
# Refer to <No-Reference quality assessment for blur image combined with re-blur range and singular value decomposition>
def cal_reblur(img):
    # Degraded image using Gaussian blur
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    metric = 0.0
    height, width = img.shape

    blur = cv.GaussianBlur(img, ksize=(9, 9), sigmaX=1, sigmaY=1)

    U_test, S_test, Vt_test = np.linalg.svd(img, full_matrices=True, compute_uv=True)
    U_ref, S_ref, Vt_ref = np.linalg.svd(blur, full_matrices=True, compute_uv=True)

    V_test = Vt_test.T
    V_ref = Vt_ref.T

    r_test = len(S_test)
    r_ref = len(S_ref)
    D = []
    for idx in range(0, min(r_test, r_ref)):
        D_i = abs(
            np.dot(U_ref[:, idx], (V_ref[:r_test, idx].T)) *
            np.dot(U_test[:, idx], (V_test[:r_test, idx].T)) /
            (np.sqrt(np.dot(U_ref[:, idx], (V_ref[:r_test, idx].T))**2))
            * (np.sqrt(np.dot(U_test[:, idx], (V_test[:r_test, idx].T))**2))
        )
        D.append(D_i)

    sum_sigma_test = cv.sumElems(S_test)
    sum_sigma_ref = cv.sumElems(S_ref)

    N = 64
    p = []  # representing re-blur feature

    p_0 = sum(S_test + S_ref)/(np.sqrt(sum(S_test ** 2)) * np.sqrt(sum(S_ref ** 2)))
    p.append(p_0)
    for j in range(1, min(r_test, r_ref)):
        p.append(D[j])


    print(p)
    print(min(r_test, r_ref))

# Haar wavelet transform in blur detection
# Refer to <Blur Detection for Digital Images Using Wavelet Transform>
# HWT performs mediocre.
pass

# <editor-fold desc = "blur region detection using svd and small-size window">
# Using svd to give a degree/metric on blurred/non-blurred image
# the best threshold = 0.75??? refer to related articles.
def cal_svd_metric(img, sv_num=10):
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv

# svd and small-size patch
def cal_blur_map(img, win_size=5, sv_num=5):
    # img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    new_img = np.zeros((img.shape[0]+win_size*2, img.shape[1]+win_size*2))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if i<win_size:
                p = win_size-i
            elif i>img.shape[0]+win_size-1:
                p = img.shape[0]*2-i
            else:
                p = i-win_size
            if j<win_size:
                q = win_size-j
            elif j>img.shape[1]+win_size-1:
                q = img.shape[1]*2-j
            else:
                q = j-win_size

            new_img[i, j] = img[p, q]

    blur_map = np.zeros((img.shape[0], img.shape[1]))
    max_sv = 0
    min_sv = 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            block = new_img[i:i+win_size*2, j:j+win_size*2]
            u, s, v = np.linalg.svd(block)  # calculate svd is time-consuming
            top_sv = np.sum(s[0:sv_num])
            total_sv = np.sum(s)
            sv_degree = top_sv/total_sv
            if max_sv < sv_degree:
                max_sv = sv_degree
            if min_sv > sv_degree:
                min_sv = sv_degree
            blur_map[i, j] = sv_degree

    blur_map = (blur_map-min_sv)/(max_sv-min_sv)

    return (1-blur_map)*255

def draw_blur_region(img, blur_map, threshold=80):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if blur_map[i, j] > threshold:
                img[i, j][1] = cv.add(img[i, j], 150)[0]
            elif(blur_map[i, j] > 10 and blur_map[i, j] < threshold):
                img[i, j][2] = cv.add(img[i, j], 150)[0]
    return img

# implement cal_blur_map and draw_blur_region
def cal_region_blur(img, win_size=5, sv_num=5, threshold=80):
    blur_map_norm = cal_blur_map(img, win_size, sv_num)
    draw_blur_region(img, blur_map_norm, threshold)
    return img
# </editor-fold>

# Blur-detector : IQA -- no-reference, threshold = [1, 5]
# Refers to <No-Reference Image Quality Assessment using Blur and Noise>
def cal_iqa(image, threshold, weight_1=1.55, weight_2=0.86):
    img = image.copy()
    if len(img.shape) == 3:
        (img, _, _) = cv.split(img)
    h, w = img.shape
    sum_diff_horizontal, sum_diff_vertical = 0, 0
    candidate_array_horizontal = np.zeros((h, w))
    candidate_array_vertical = np.zeros((h, w))
    edge_horizontal = np.zeros((h, w))
    edge_vertical = np.zeros((h, w))

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            diff_horizontal = abs(int(img[y + 1, x]) - int(img[y - 1, x]))
            diff_vertical = abs(int(img[y, x + 1]) - int(img[y, x - 1]))

            sum_diff_horizontal += diff_horizontal
            sum_diff_vertical += diff_vertical

    diff_horizontal_mean = sum_diff_horizontal / (w * h)
    diff_vertical_mean = sum_diff_vertical / (w * h)

    blur_count = 1
    sum_BR_horizontal, sum_BR_vertical = 0, 0
    for x in range(1, w - 1):
        for y in range(1, h - 1):

            diff_horizontal = abs(int(img[y + 1, x]) - int(img[y - 1, x])) + 1
            candidate_array_horizontal[y][x] = diff_horizontal if diff_horizontal > diff_horizontal_mean else 0

            diff_vertical = abs(int(img[y, x + 1]) - int(img[y, x - 1])) + 1
            candidate_array_vertical[y][x] = diff_horizontal if diff_vertical > diff_vertical_mean else 0

            BR_horizontal = 2 * abs(int(img[y, x]) - diff_horizontal) / diff_horizontal
            BR_vertical = 2 * abs(int(img[y, x]) - diff_vertical) / diff_vertical

            sum_BR_horizontal += BR_horizontal
            sum_BR_vertical += BR_vertical

            if (max(BR_horizontal, BR_vertical) < threshold):
                blur_pixel = 1
                blur_count += 1
            else:
                blur_pixel = 0

    edge_horizontal_count, edge_vertical_count = 0, 0
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if candidate_array_horizontal[y][x] > candidate_array_horizontal[y + 1][x] and candidate_array_horizontal[y][x] > candidate_array_horizontal[y - 1][x]:
                edge_horizontal[y][x] = 1
                edge_horizontal_count += 1
            else:
                edge_horizontal[y][x] = 0

            if candidate_array_vertical[y][x] > candidate_array_vertical[y + 1][x] and candidate_array_vertical[y][x] > candidate_array_vertical[y - 1][x]:
                edge_vertical[y][x] = 1
                edge_vertical_count += 1
            else:
                edge_vertical[y][x] = 0

    blur_mean = max(sum_BR_horizontal, sum_BR_vertical) / blur_count
    blur_ratio = blur_count / (edge_horizontal_count + edge_vertical_count)
    metric = 1 - (weight_1 * blur_mean + weight_2 * blur_ratio)
    return metric

# Draw the result image using IQA function.
def draw_blur_detection(img, inputTemplates, thresholdBlurRate=0.5):
    inputImage = img.copy()
    result_blur_detection = img.copy()
    for index in range(0, len(inputTemplates)):
        x = inputTemplates[index].x
        y = inputTemplates[index].y
        height = inputTemplates[index].height
        width = inputTemplates[index].width
        label = inputTemplates[index].content

        crop = inputImage[y: y + height, x: x + width]
        top_left = (x, y)
        bottom_right = (x + width, y + height)

        metric_blur_detection = cal_iqa(image=crop, threshold=1.0)  # this threshold is for IQA function itself, fixed by 1.0

        if metric_blur_detection < -thresholdBlurRate*1.0e5:  # threshold needs to be modified
            cv.rectangle(result_blur_detection, top_left, bottom_right, (0, 0, 230), 1)
        else:
            cv.rectangle(result_blur_detection, top_left, bottom_right, (0, 230, 0), 1)

    return result_blur_detection

# TODO: use the multi-scale perception
def cal_multi_scale_perception(img):
    # generate feature map
    nStates = 2

# TODO: train the local filter and output the result
def cal_local_filter(img):
    pass

# Normalize the metric, m0 -> 100
def fun_norm_metric(m0, m1, m2, m3):
    norm_m1 = m1 / m0 * 100
    norm_m2 = m2 / m0 * 100
    norm_m3 = m3 / m0 * 100
    if norm_m1 <= 100:
        norm_m1 = 100 - norm_m1 + 100
    elif norm_m2 <= 100:
        norm_m2 = 100 - norm_m2 + 100
    elif norm_m3 <= 100:
        norm_m3 = 100 - norm_m3 + 100
    print(norm_m1, ", ", norm_m2, ", ", norm_m3)


def main():
    img = cv.imread("image\\snipped_02.png")
    img_binary, img_blur = pp.convert_to_binary(img=img, kernel_value=5)
    # var = variance_of_laplacian(img)
    # var_blur = variance_of_laplacian(img_blur)
    img_motion_x = cv.blur(img, ksize=(15, 1), anchor=(-1, -1), borderType=4)
    img_motion_y = cv.blur(img, ksize=(1, 15), anchor=(-1, -1), borderType=4)

    # img_motion_x = cv.cvtColor(img_motion_x, cv.COLOR_BGR2GRAY)
    # img_motion_y = cv.cvtColor(img_motion_y, cv.COLOR_BGR2GRAY)

    # nrss_img = str(round(cal_nrss(img), 5))
    # cv.putText(img, nrss_img, (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    function_name = 'energy'

    img = cv.imread("image\\snipped_02.png")
    flat = cv.imread('standard.png')  # flat is one style of blurred image.
    region_blurred0 = cv.imread('image\\blurred.png')
    region_blurred1 = cv.imread('image\\region_blur.png')

    t0 = ps.TemplateBox(14, 12, 38, 40, "")  # M
    t1 = ps.TemplateBox(49, 11, 35, 40, "")  # O
    t2 = ps.TemplateBox(83, 12, 33, 40, "")  # D
    t3 = ps.TemplateBox(115, 11, 28, 43, "")  # E
    t4 = ps.TemplateBox(143, 10, 30, 42, "")  # L
    t5 = ps.TemplateBox(14, 60, 31, 44, "")  # P
    t6 = ps.TemplateBox(45, 59, 32, 45, "")  # O
    t7 = ps.TemplateBox(76, 60, 44, 44, "")  # W
    t8 = ps.TemplateBox(119, 60, 28, 44, "")  # E
    t9 = ps.TemplateBox(147, 59, 32, 44, "")  # R
    listTemplate = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]

    res = draw_blur_detection(region_blurred1, listTemplate, thresholdBlurRate=0.5)
    cv.imshow('blur with pre-segment', res)

    res_area = cal_region_blur(region_blurred1, threshold=55)
    cv.imshow('blur without pre-segment', res_area)

    # region_blurred = cv.imread('image\\baseline_region_blur.png')

    # m0 = cal_iqa(img, threshold)
    # m1 = cal_iqa(blur, threshold)
    # m2 = cal_iqa(img_motion_x, threshold)
    # m3 = cal_iqa(img_motion_y, threshold)
    #
    #
    # fun_norm_metric(m0, m1, m2, m3)

    # cv.putText(img, str(round(metric_img, 3)), (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # cv.putText(blur, str(round(metric_blur, 3)), (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # cv.putText(img_motion_x, str(round(metric_motion_x, 3)), (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # cv.putText(img_motion_y, str(round(metric_motion_y, 3)), (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # cv.imshow('img', img)
    # cv.imshow('blur', blur)
    # cv.imshow('motion x', img_motion_x)
    # cv.imshow('motion y', img_motion_y)
    #
    # cv.imwrite('result\\' + function_name + '_origin.png', img)
    # cv.imwrite('result\\' + function_name + '_blur.png', blur)
    # cv.imwrite('result\\' + function_name + '_motion_x.png', img_motion_x)
    # cv.imwrite('result\\' + function_name + '_motion_y.png', img_motion_y)

if __name__ == '__main__':
    main()
    cv.waitKey(0)
