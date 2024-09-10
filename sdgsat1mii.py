# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/16 9:50
@FileName: sdgsat1mii.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from osgeo import gdal
import glob
from lxml import etree
import os
import numpy as np
from stretch import linearStretch


def read_lt(file):
    src_ds = gdal.Open(file.get('file'))
    im_width = src_ds.RasterXSize  # 栅格矩阵的列数
    im_height = src_ds.RasterYSize  # 栅格矩阵的行数
    bound = file.get('area')  # [开始行，结束行，开始列，结束列]
    dn = src_ds.ReadAsArray(bound[2], bound[0], bound[3] - bound[2], bound[1] - bound[0]).transpose(1, 2, 0) * 1.
    # from xml.etree import ElementTree as ET
    # tree = ET.parse(file)
    # time = tree.find("SatelliteInfo").find("CenterTime").find("Acamera").text
    # saa = tree.find("SatelliteInfo").find("SolarAzimuth").text
    # sza = tree.find("SatelliteInfo").find("SolarZenith").text
    # roll = tree.find("SatelliteInfo").find("RollSatelliteAngle").text
    # pitch = tree.find("SatelliteInfo").find("PitchSatelliteAngle").text
    # yaw = tree.find("SatelliteInfo").find("YawSatelliteAngle").text
    infile = file.get('file')
    dirname = os.path.dirname(infile)
    basename = os.path.basename(infile)
    name_id = "_".join(basename.split("_")[0:7])
    calib_file_path = glob.glob(dirname + os.sep + name_id + "*.calib.xml")[0]
    meta_file_path = glob.glob(dirname + os.sep + name_id + "*.meta.xml")[0]
    root = etree.parse(calib_file_path)
    gains = np.array(
        [float(root.find("./RADIOMETRIC_CALIBRATION/MII/VERSION/RADIANCE_GAIN_BAND_" + str(i + 1)).text) for i in
         range(7)]) / 10
    bias = np.array(
        [float(root.find("./RADIOMETRIC_CALIBRATION/MII/VERSION/RADIANCE_BIAS_BAND_" + str(i + 1)).text) for i in
         range(7)])
    lt = dn * gains.reshape(1, 1, -1) + bias.reshape(1, 1, -1)
    # wave, f0 = read_f0_bands()
    root1 = etree.parse(meta_file_path)
    sza = float(root1.find("SatelliteInfo/SolarZenith").text)
    # lt / np.cos(np.deg2rad(sza))
    return lt*10, sza  # / np.cos(np.deg2rad(sza))


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
    # gains, bias = self.cali(infile=self.image_file)
    # gains = np.array([0.051560133, 0.036241353, 0.023316835, 0.015849666, 0.016096381, 0.019719039, 0.013811458])
    # bias = np.array([0, 0, 0, 0, 0, 0, 0])
    src_ds = gdal.Open(file)
    # im_width = src_ds.RasterXSize  # 栅格矩阵的列数
    # im_height = src_ds.RasterYSize  # 栅格矩阵的行数
    # data = src_ds.ReadAsArray(0, 0, im_width, im_height)
    [band_r, band_g, band_b] = band_rgb

    srcband1 = src_ds.GetRasterBand(band_r)
    red = skimage.measure.block_reduce(srcband1.ReadAsArray() * 1., block_size=(resize, resize), func=np.nanmean,
                                       cval=np.nan)
    red = linearStretch(red, 0.02)
    srcband2 = src_ds.GetRasterBand(band_g)
    green = skimage.measure.block_reduce(srcband2.ReadAsArray() * 1., block_size=(resize, resize), func=np.nanmean,
                                         cval=np.nan)
    green = linearStretch(green, 0.02)
    srcband3 = src_ds.GetRasterBand(band_b)
    blue = skimage.measure.block_reduce(srcband3.ReadAsArray() * 1., block_size=(resize, resize), func=np.nanmean,
                                        cval=np.nan)
    blue = linearStretch(blue, 0.02)

    return np.dstack([red, green, blue])