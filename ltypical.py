# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/19 20:08
@FileName: ltypical.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830


"""
import numpy as np


def l(sensor_id):
    match sensor_id:
        case "sdgsat1mii":
            typicalR_sza10 = np.array([10.83, 9.02, 6.27, 3.7, 1.7, 0.92, 0.62]) * 10
            typicalR_sza30 = np.array([9.87, 8.53, 6.01, 3.58, 1.62, 0.84, 0.56]) * 10
            typicalR_sza45 = np.array([8.47, 7.16, 5.02, 3.04, 1.36, 0.67, 0.45]) * 10
            typicalR_sza70 = np.array([5.12, 4.55, 3.32, 1.99, 0.96, 0.49, 0.31]) * 10
        case "sentinel2bmsi":
            typicalR_sza10 = np.array([8.81, 6.39, 3.42, 1.61, 1.35, 1.1, 0.9, 0.69, 0.56, 0.48, 0.06, 0.01]) * 10
            typicalR_sza30 = np.array([8.4, 6.12, 3.33, 1.53, 1.25, 1.01, 0.82, 0.62, 0.51, 0.43, 0.05, 0.01]) * 10
            typicalR_sza45 = np.array([7.02, 5.12, 2.83, 1.28, 1.03, 0.81, 0.65, 0.5, 0.41, 0.35, 0.04, 0.01]) * 10
            typicalR_sza70 = np.array([4.5, 3.37, 1.85, 0.92, 0.76, 0.6, 0.48, 0.35, 0.28, 0.24, 0.02, 0.0]) * 10
        case "landsat8oli":
            typicalR_sza10 = np.array([8.78, 6.9, 3.41, 1.69, 0.56, 0.06, 0.01]) * 10
            typicalR_sza30 = np.array([8.38, 6.61, 3.31, 1.61, 0.51, 0.05, 0.01]) * 10
            typicalR_sza45 = np.array([6.99, 5.51, 2.81, 1.35, 0.41, 0.04, 0.0]) * 10
            typicalR_sza70 = np.array([4.49, 3.62, 1.84, 0.95, 0.28, 0.02, 0.0]) * 10
        case "landsat9oli2":
            typicalR_sza10 = np.array([8.77, 6.91, 3.42, 1.69, 0.56, 0.06, 0.01]) * 10
            typicalR_sza30 = np.array([8.37, 6.62, 3.32, 1.61, 0.51, 0.05, 0.01]) * 10
            typicalR_sza45 = np.array([6.99, 5.52, 2.82, 1.35, 0.41, 0.04, 0.0]) * 10
            typicalR_sza70 = np.array([4.49, 3.63, 1.84, 0.95, 0.28, 0.02, 0.0]) * 10
        case "sentinel3bolci":
            typicalR_sza10 = np.array([
                10.83, 10.27, 8.77, 6.41, 5.39, 3.35, 2.2, 1.62, 1.56, 1.51, 1.31, 1.01, 0.98, 0.97, 0.95, 0.9, 0.56,
                0.54, 0.53, 0.48, 0.39]) * 10
            typicalR_sza30 = np.array([
                9.87, 9.47, 8.38, 6.17, 5.18, 3.25, 2.12, 1.54, 1.47, 1.41, 1.22, 0.93, 0.9, 0.88, 0.87, 0.82, 0.51,
                0.49, 0.48, 0.44, 0.36]) * 10
            typicalR_sza45 = np.array([
                8.47, 8.07, 6.99, 5.14, 4.35, 2.76, 1.79, 1.28, 1.22, 1.17, 1.0, 0.73, 0.71, 0.7, 0.69, 0.65, 0.41, 0.4,
                0.38, 0.35, 0.28]) * 10
            typicalR_sza70 = np.array([
                5.12, 4.95, 4.49, 3.43, 2.89, 1.8, 1.2, 0.92, 0.89, 0.86, 0.73, 0.54, 0.52, 0.52, 0.51, 0.48, 0.28,
                0.28, 0.27, 0.25, 0.2]) * 10
        case "sentinel3aolci":
            typicalR_sza10 = np.array([
                10.84, 10.28, 8.77, 6.4, 5.39, 3.35, 2.19, 1.62, 1.56, 1.51, 1.31, 1.01, 0.98, 0.97, 0.95, 0.9, 0.56,
                0.54, 0.53, 0.48, 0.39]) * 10
            typicalR_sza30 = np.array([
                9.88, 9.48, 8.38, 6.16, 5.18, 3.25, 2.11, 1.54, 1.47, 1.41, 1.22, 0.92, 0.89, 0.88, 0.87, 0.82, 0.51,
                0.49, 0.48, 0.44, 0.36]) * 10
            typicalR_sza45 = np.array([
                8.48, 8.08, 6.99, 5.14, 4.35, 2.76, 1.79, 1.28, 1.22, 1.17, 0.99, 0.73, 0.71, 0.7, 0.69, 0.65, 0.41,
                0.4, 0.38, 0.35, 0.28]) * 10
            typicalR_sza70 = np.array([
                5.12, 4.95, 4.5, 3.42, 2.88, 1.8, 1.2, 0.92, 0.89, 0.85, 0.73, 0.54, 0.52, 0.52, 0.51, 0.48, 0.28, 0.28,
                0.27, 0.25, 0.2]) * 10

    typicalR_sza_all = [typicalR_sza10, typicalR_sza30, typicalR_sza45, typicalR_sza70]

    return typicalR_sza_all



"""
文献
Hu, Chuanmin, Lian Feng, Zhongping Lee, Curtiss O. Davis, Antonio Mannino, Charles R. McClain, and Bryan A. Franz. 
"Dynamic range and sensitivity requirements of satellite ocean color sensors: learning from the past." 
Applied Optics 51, no. 25 (2012): 6045-6062.
"""

"""
计算典型辐亮度下的的校正系数
"""


import pandas as pd
import scipy.interpolate as interpolate


def interpolate_fun(x, y, x_new=None):
    """
    插值函数
    """
    if type(x) is np.ndarray:
        x = x
    else:
        # dataframe
        x = x.values
    if type(y) is np.ndarray:
        y = y
    else:
        y = y.values

    f = interpolate.interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    y_new = f(x_new)
    x_new = x_new
    return [x_new, y_new]


def calculate_band_average(reference_wave=None, reference_spectrum=None, spectrum_response_function=None):
    """
    插值
    积分
    """
    reference_wave = reference_wave
    reference_response = reference_spectrum

    srf_wave = spectrum_response_function.iloc[:, 0]
    # 波段数：
    bands = spectrum_response_function.shape[1] - 1
    band_values = []
    for i in range(bands):
        srf_band = spectrum_response_function.iloc[:, 1 + i]
        wave, solar_spectrum_new = interpolate_fun(reference_wave, reference_response, x_new=srf_wave)
        band_values_ = np.nansum(solar_spectrum_new * srf_band) / np.nansum(srf_band)
        band_values.append(band_values_)
    return band_values


class TypicalRadiance(object):
    def __init__(self):
        # 光谱响应函数
        self.rsr_infile = r'D:\researchProject_lwk\code\atmoscorr\AC_l2gen\RSR\RSR.xlsx'
        self.sensorid= "landsat9oli2" #"sentinel3bolci"#"landsat8oli" #"sentinel2bmsi"
        print(self.sensorid)

    def run_main(self):

        # lambda
        # center_wavelength = [401, 438, 495, 553, 657, 776, 854]
        target_rsr = self.target_rsr()
        center_wavelength = target_rsr.columns.tolist()
        self.reference()
        typical_0 = calculate_band_average(reference_wave=self.wave, reference_spectrum=self.typicalL_sza10,
                                           spectrum_response_function=target_rsr)
        typical_1 = calculate_band_average(reference_wave=self.wave, reference_spectrum=self.typicalL_sza30,
                                           spectrum_response_function=target_rsr)
        typical_2 = calculate_band_average(reference_wave=self.wave, reference_spectrum=self.typicalL_sza45,
                                           spectrum_response_function=target_rsr)
        typical_3 = calculate_band_average(reference_wave=self.wave, reference_spectrum=self.typicalL_sza70,
                                           spectrum_response_function=target_rsr)

        print("band", center_wavelength)
        print("sza10", [round(i, 2) for i in typical_0])
        print("sza30", [round(i, 2) for i in typical_1])
        print("sza45", [round(i, 2) for i in typical_2])
        print("sza70", [round(i, 2) for i in typical_3])

    def target_rsr(self):
        rsr = pd.read_excel(io=self.rsr_infile, sheet_name=self.sensorid, header=0, index_col=None)
        # wavelength=rsr['Wavelength (nm)']
        # rsr = rsr[(rsr['Wavelength (nm)'] >= 370) & (rsr['Wavelength (nm)'] <= 980)]
        return rsr

    def reference(self):
        """
        MODIS
        文献
        Hu, Chuanmin, Lian Feng, Zhongping Lee, Curtiss O. Davis, Antonio Mannino, Charles R. McClain, and Bryan A. Franz.
        "Dynamic range and sensitivity requirements of satellite ocean color sensors: learning from the past."
        Applied Optics 51, no. 25 (2012): 6045-6062.
        """
        self.wave = np.array([412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748, 859, 869, 1240, 1640, 2130])
        self.typicalL_sza10 = np.array(
            [10.27, 8.75, 7.8, 6.52, 4.36, 3.74, 3.45, 1.72, 1.61, 1.53, 1.04, 0.55, 0.56, 0.14, 0.056, 0.015])
        self.typicalL_sza30 = np.array(
            [9.47, 8.38, 7.43, 6.28, 4.17, 3.67, 3.35, 1.65, 1.53, 1.43, 0.95, 0.5, 0.51, 0.12, 0.045, 0.011])
        self.typicalL_sza45 = np.array(
            [8.07, 6.98, 6.19, 5.23, 3.55, 3.13, 2.85, 1.39, 1.27, 1.19, 0.75, 0.4, 0.41, 0.086, 0.031, 0.008])
        self.typicalL_sza70 = np.array(
            [4.95, 4.5, 4.01, 3.49, 2.33, 2.05, 1.85, 0.96, 0.92, 0.87, 0.56, 0.27, 0.29, 0.058, 0.018, 0.004])


if __name__ == '__main__':
    TypicalRadiance().run_main()