# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/22 14:42
@FileName: input_data.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os


def data():
    #  [开始行，结束行，开始列，结束列]

    filedir = r"G:\SDGsat\radiometeric\SNR\SCS\V2\LC08_L1TP_117048_20240109_20240122_02_T1"
    a5 = {'file': filedir + os.sep + "LANDSAT8_OLI.20240109T022255.L2.OC.nc",
          "area": [2913, 4152, 1716, 2759],
          "sensor_id": "landsat8oli"}

    filedir = r"G:\SDGsat\radiometeric\SNR\SCS\V2\S2B_MSIL1C_20240105T023109_N0510_R046_T50QQE_20240105T034253.SAFE"
    a6 = {'file': filedir + os.sep + "S2B_MSI.20240105T023109.L2.OC.nc",
          "area": [3910, 5043, 352, 1559],
          "sensor_id":"sentinel2bmsi"}

    # landsat9 oli
    filedir = r"G:\SDGsat\radiometeric\SNR\SCS\L9OLI\LC09_L1GT_117051_20221127_20230320_02_T2"
    a5 = {
        'file': filedir + os.sep + "LANDSAT9_OLI.20221127T022426.L2.OC.nc",
        "area": [4038, 4838, 1706, 2832],
        "sensor_id": "landsat9oli2"}

    return [a5]

