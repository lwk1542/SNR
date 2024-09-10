# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/17 14:30
@FileName: seadas.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import numpy as np
import h5py


def read_lt(file):
    ds = h5py.File(file.get('file'), mode="r")
    bound = file.get('area')  # [开始行，结束行，开始列，结束列]
    ds_sub1 = ds["geophysical_data"]
    geophy = "Lt_"
    wavelength = sorted([int(i.replace(geophy, "")) for i in ds_sub1.keys() if geophy in i], reverse=False)
    data = np.empty(shape=(bound[1] - bound[0], bound[3] - bound[2], wavelength.__len__()))
    for i, band in enumerate(wavelength):
        _ = ds_sub1[geophy + str(band)]
        fillvalue = _.attrs["_FillValue"][0]
        valid_max = _.attrs["valid_max"][0]
        valid_min = _.attrs["valid_min"][0]
        value = _[bound[0]:bound[1], bound[2]:bound[3]]
        value[(value > valid_max) | (value < valid_min) | (value == fillvalue)| (value < 0)] = np.nan
        data[:, :, i] = value
    sza = ds["geophysical_data/solz"]
    fillvalue = sza.attrs["_FillValue"][0]
    valid_max = sza.attrs["valid_max"][0]
    valid_min = sza.attrs["valid_min"][0]
    sza_ = sza[bound[0]:bound[1], bound[2]:bound[3]]*1.
    sza_[(sza_ > valid_max) | (sza_ < valid_min) | (sza_ == fillvalue)] = np.nan
    scale_factor = sza.attrs["scale_factor"][0]
    add_offset = sza.attrs["add_offset"][0]
    sza_ = sza_*scale_factor+add_offset
    print(np.nanmax(sza_))

    return data, sza_.reshape(sza_.shape[0], sza_.shape[1], 1)  #/ np.cos(np.deg2rad(sza_)).reshape(sza_.shape[0], sza_.shape[1], 1)


def read_rgb(file, band_rgb, resize:int):
    import skimage.measure
    """
    标注使用的区域导出为jpg
    Args:
        infile:
        loc:
        m:
    Returns:
    """
    ds = h5py.File(file, mode="r")
    ds_sub1 = ds["geophysical_data"]
    geophy = "Lt_"
    wavelength = sorted([int(i.replace(geophy, "")) for i in ds_sub1.keys() if geophy in i], reverse=False)
    wave_rgb = [wavelength[i] for i in band_rgb]
    for i, band in enumerate(wave_rgb):
        _ = ds_sub1[geophy + str(band)]
        fillvalue = _.attrs["_FillValue"][0]
        valid_max = _.attrs["valid_max"][0]
        valid_min = _.attrs["valid_min"][0]
        value = _[()]
        value[(value > valid_max) | (value < valid_min) | (value == fillvalue)| (value < 0)] = np.nan
        # perc = np.nanpercentile(value, [2, 98])
        # value[(value > perc[1]) | (value < perc[0])] = np.nan
        if i == 0:
            data = value
        else:
            data = np.dstack([value, data])
    data = skimage.measure.block_reduce(data, block_size=(resize, resize, 1), func=np.nanmean, cval=np.nan)
    return data


if __name__ == '__main__':
    filedir = r"G:\SDGsat\radiometeric\SNR\landsat8oli"
    a2 = {'file': filedir + os.sep + "LANDSAT8_OLI.20230922T210018.L2.OC.nc",
          "area": [5018, 5876, 3264, 4222]}
    read_lt(a2)