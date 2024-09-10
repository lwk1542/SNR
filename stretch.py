# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/16 9:54
@FileName: stretch.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np


def hist_calc(img, ratio):
    # img[img < 0]=np.nan
    # 调用Numpy实现灰度统计
    valid_data = img[~np.isnan(img)]
    hist, bins = np.histogram(valid_data, bins=100)
    total_pixels = np.nansum(hist)
    # 计算获得ratio%所对应的位置，
    # 这里ratio为0.02即为2%线性化，0.05即为5%线性化
    min_index = int(ratio * total_pixels)
    max_index = int((1 - ratio) * total_pixels)
    min_gray = 0
    max_gray = 0
    # 统计最小灰度值(A)
    sum1 = 0
    for i in range(hist.__len__()):
        sum1 = sum1 + hist[i]
        if sum1 > min_index:
            min_gray = i
            break
    # 统计最大灰度值(B)
    sum2 = 0
    for i in range(hist.__len__()):
        sum2 = sum2 + hist[i]
        if sum2 > max_index:
            max_gray = i
            break
    return min_gray, max_gray


def linearStretch(array, ratio):
    # 获取原图除去2%像素后的最小、最大灰度值(A、B)
    old_min, old_max = hist_calc(array, ratio)
    # 对原图中所有小于或大于A、B的像素都赋为A、B
    array[array < old_min] = old_min
    array[array > old_max] = old_max
    #         print('old min = %d,old max = %d' % (np.nanmin(array), np.nanmax(array)))
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    return (array - array_min) / (array_max - array_min)