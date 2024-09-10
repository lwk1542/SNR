# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/2 9:09
@FileName: SNR_main.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""

import os
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter
import input_data
import pandas as pd


class SNR(object):
    def __init__(self):
        # self.sensor_id = "sdgsat1mii"
        pass

    def draw2(self, data: np.ndarray, out_dir, image_file):
        """
        检查边框选的是否合理，判断标准是没有明显的趋势
        Returns
        -------
        """
        fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.5))
        color = ['black', 'blue', 'red', 'green']
        for i, d in enumerate([data[0, :, 0], data[-1, :, 0], data[:, 0, 0], data[:, -1, 0]]):
            x = np.arange(d.size)
            ax.plot(x, d, linestyle="-", color=color[i])
        ax.set_ylabel('DN')
        plt.subplots_adjust(bottom=0.14, right=0.99, left=.24, top=0.93, wspace=0.4, hspace=0.05)
        figname = out_dir + os.sep + os.path.basename(image_file)[0:-3] + '_localSignal.jpg'
        plt.savefig(figname, dpi=300)
        # plt.show()
        plt.close()

    def roi(self, file, out_dir, image_file):
        image = file.get('file')
        loc = file.get('area')
        _, read_rgb = self.package_match()
        resize = int(1000 / self.sensor_set["resolution"])
        arr = read_rgb(image, self.sensor_set["b_rgb"], resize)
        arr = (arr - np.nanmin(arr)) * 255 / (np.nanmax(arr) - np.nanmin(arr))
        loc = [int(i / resize) for i in loc]
        cv2.rectangle(arr, (loc[2], loc[0]), (loc[3], loc[1]), (0, 0, 255), 2)
        # 标注文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'here'
        cv2.putText(arr, text, (np.min(loc[1]), np.min(loc[0]) - 5), font, 2, (0, 255, 0), 3)
        cv2.imwrite(out_dir + os.sep + os.path.basename(image_file)[0:-3] + '_studyarea.jpg', arr)

    def drawGraph(self, hists_multi, snr_multi, window, out_dir, image_file):
        '''
        作图
        '''
        font = {'family': 'Times New Roman',
                'color': 'black',
                'weight': 'normal',
                'size': 9
                }
        matplotlib.rcParams["font.family"] = "Times New Roman"  # 全局times new roamn
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'

        for j, wave in enumerate(hists_multi):
            fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.5))
            print(self.wavelength[j])
            snr = snr_multi[j]
            hists = hists_multi[j]
            for i, [bin_temp, hist_temp] in enumerate(hists):
                x = []
                max_snr = []
                max_snr.append(np.nanmax(hist_temp))
                for k in range(len(bin_temp) - 1):
                    x.append((bin_temp[k] + bin_temp[k + 1]) / 2)
                ax.plot(x, hist_temp, label=str(window[i][0]) + "×" + str(window[i][1]), linewidth=0.5)  # ,color="red"
            ax.plot([snr, snr], [0, np.nanmax(max_snr) * 1.4], linestyle="dashed", color="black")
            # ax.set_xlim(0, np.max(x))
            ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
            ax.spines['bottom'].set_linewidth(1.25)
            ax.spines['left'].set_linewidth(1.25)
            ax.spines['right'].set_linewidth(1.25)
            ax.spines['top'].set_linewidth(1.25)
            ax.tick_params(labelright=None)
            f = plt.gcf()
            f.text(0.63, 0.77, u'band:' + str(self.wavelength[j]) + 'nm', fontdict=font, color='black')
            f.text(0.63, 0.67, u'SNR:' + str(int(snr)), fontdict=font, color='black')
            f.text(0.015, 0.43, u'#Number of pixels', fontdict=font, color='black', rotation=90)
            f.text(0.52, 0.02, u'SNR', fontdict=font, color='black')  # $$ {}的使用 late\
            ax.legend(loc=1, borderaxespad=0., fontsize=6)
            plt.subplots_adjust(bottom=0.14, right=0.95, left=.17, top=0.93, wspace=0.4, hspace=0.05)
            figname = out_dir + os.sep + os.path.basename(image_file)[0:-3] + "band_" + str(
                int(self.wavelength[j])) + '_SNR.png'
            plt.savefig(figname, dpi=300)
            # plt.show()
            plt.close()

    def package_match(self):
        match self.sensor_id:
            case 'sdgsat1mii':
                from sdgsat1mii import read_lt, read_rgb
            case "landsat8oli" | "landsat9oli2" | "sentinel3bolci" | "sentinel2bmsi":
                from seadas import read_lt, read_rgb

        return read_lt, read_rgb

    def read_image(self, file):
        read_lt, _ = self.package_match()
        lt, sza = read_lt(file)
        self.wavelength, F0 = self.read_f0_bands()
        # rho = lt / np.array(F0).reshape(1, 1, -1) / np.cos(np.deg2rad(sza))
        return lt, sza, np.array(F0).reshape(1, 1, -1)[:-2]

    def read_f0_bands(self):
        # 根据指定的传感器获取查找表路径
        print('sensorID: ' + self.sensor_id)
        _ = r"D:\researchProject_lwk\code\atmoscorr\AC_l2gen/share"
        match self.sensor_id:
            case 'hy1ccocts':
                lut_path = _ + os.sep + 'hy1ccocts'
            case 'hy1dcocts':
                lut_path = _ + os.sep + 'hy1dcocts'
            case 'fy3dmersi':
                lut_path = _ + os.sep + "fy3dmersi"
            case "sdgsat1mii":
                lut_path = _ + os.sep + "sdgsat1mii"
            case 'seawifsphd':
                lut_path = _ + os.sep + "seawifsphd"
            case "landsat8oli":
                lut_path = _ + os.sep + "landsat8oli"
            case "landsat9oli2":
                lut_path = _ + os.sep + "landsat9oli2"
            case "sentinel2bmsi":
                lut_path = _ + os.sep + "sentinel2bmsi"
            case "sentinel3bolci":
                lut_path = _ + os.sep + "sentinel3bolci"
            case _:
                print("Error: Can not identify satellite sensor ID for obtaining look-up table... ")

        f = open(lut_path + os.sep + "msl12_sensor_info.dat")
        lines = f.readlines()
        # filter note
        lines = [i for i in lines if not "#" in i.lower()]
        self.wavelength = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "lambda(" in i.lower()]
        F0 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "f0(" in i.lower()]
        return self.wavelength, F0

    def slid_window(self, data, window=list[int, int]):
        # scipy.ndimage.generic_filter()跑的比较慢
        # std = scipy.ndimage.generic_filter(data, np.nanstd, size=(window[0], window[1], 1))
        # avg = scipy.ndimage.generic_filter(data, np.nanmean, size=(window[0], window[1], 1))
        # cv = avg / std
        # for i in range(cv.shape[2]):
        #     cvi = cv[:, :, i]
        #     low, high = np.percentile(cvi, [5, 95])
        #     cvi[(cvi < low) | (cvi > high)] = np.nan
        #     cv[:, :, i] = cvi

        cv = np.empty_like(data)
        for i in range(cv.shape[2]):
            # 计算滑动窗口的均值
            window_means = uniform_filter(data[:, :, i], size=window[0])
            # 计算滑动窗口的标准差
            window_squared_means = uniform_filter(data[:, :, i] ** 2, size=window[0])
            window_stds = np.sqrt(window_squared_means - window_means ** 2)
            cv[:, :, i] = window_means / window_stds
            low, high = np.percentile(cv[:, :, i], [2, 98])
            cv[:, :, i][(cv[:, :, i] < low) | (cv[:, :, i] > high)] = np.nan

        return cv

    def snr_cali(self, cv):
        snr_multi = ()
        hist_multi = ()
        for i in range(self.wavelength.__len__()):  # 分波段
            mu = ()
            hists = ()
            for j in range(self.windows.__len__()):
                temp = cv[j][:, :, i]
                low, high = np.nanpercentile(temp, [2, 98])
                temp[(temp < low) | (temp > high)] = np.nan
                cv_filter = temp.flatten()
                cv_filter = cv_filter[~np.isnan(cv_filter)]
                # hist, bins = np.histogram(cv_filter, bins=bins_, range=range_)
                # hists = hists + (hist,)
                [hist, bins] = np.histogram(cv_filter, bins=100)
                hists = hists + ([bins, hist],)
                (mu0, sigma0) = stats.norm.fit(cv_filter)
                mu = mu + (mu0,)
            snr = np.mean(mu)
            snr_multi = snr_multi + (snr,)
            hist_multi = hist_multi + (hists,)
        return hist_multi, snr_multi

    def data(self):
        # input_data.data()
        return input_data.data()

    def setting(self, sensorid):
        set = {"landsat8oli": {"b_rgb": [3, 2, 1], "resolution": 30},
               "landsat9oli2": {"b_rgb": [3, 2, 1], "resolution": 30},
               "sdgsat1mii": {"b_rgb": [4, 3, 2], "resolution": 10},
               "sentinel2bmsi": {"b_rgb": [3, 2, 1], "resolution": 10},
               "sentinel3bolci": {"b_rgb": [12, 5, 2], "resolution": 300}}
        return set[sensorid]

    def main(self):

        reference_sensor_resolution = 30  # 以sentinel-3 OLCI为参考
        window_size = [3, 5, 7, 9, 11]
        # self.windows = [[7, 7], [11, 11], [15, 15], [19, 19]]
        # self.windows = [[3, 3], [5, 5], [7, 7], [9, 9]]
        files = self.data()
        self.sensor_id = files[0].get("sensor_id") #"sentinel3bolci"  # "sentinel2bmsi"  # "landsat8oli" # "sdgsat1mii"
        self.sensor_set = self.setting(sensorid=self.sensor_id)
        print(self.sensor_id)

        # 当有参考影像时，将窗口大小调整到与分辨率相适应，以保证窗口的空间距离是一致的，参考影像的分辨率需要低于测试数据。
        window_scale = int(round(reference_sensor_resolution / self.sensor_set["resolution"]))
        self.windows = [[i * window_scale, i * window_scale] for i in window_size]

        # print(self.sensor_id)
        # 不同角度下的典型辐亮度

        from ltypical import l
        typicalR_sza_all = l(sensor_id=self.sensor_id)
        sza_option = np.array([10, 30, 45, 70])

        dict = {}
        for j, file in enumerate(files):
            sub_dict = {}
            image_file = file.get('file')
            print(os.path.basename(image_file), self.sensor_id)
            out_dir = os.path.dirname(image_file) + os.sep + "SNR"
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            toa, sza, F0 = self.read_image(file)
            sub_dict["ifile"] = image_file
            sub_dict["odir"] = out_dir
            sub_dict["toa"] = toa
            sub_dict["sza"] = sza
            sub_dict["F0"] = F0
            typical_ind = np.argmin(np.abs(np.nanmean(sub_dict["sza"]) - sza_option))
            Lty_temp = typicalR_sza_all[typical_ind]
            scale_ltypi = Lty_temp.reshape(1, 1, -1) / np.nanmean(toa, axis=(0, 1))
            # scale_ltypi将该区域的辐亮度的调整因子，该因子直接乘以SNR就是传感器在典型辐亮度水平下的SNR
            sub_dict["scale_ltypi"] = scale_ltypi
            print("sza:{0}, scale_ltypi:{1}".format(np.mean(sza), scale_ltypi))
            dict[file.get("ifile")] = sub_dict
            self.roi(file, out_dir=out_dir, image_file=image_file)
            self.draw2(data=toa, out_dir=out_dir, image_file=image_file)

            cv = ()
            cv0 = ()
            for k, window in enumerate(self.windows):
                for j, key in enumerate(dict):
                    sub_dict = dict[key]
                    # scale_ltypi将该区域的辐亮度的调整因子，该因子直接乘以SNR就是传感器在典型辐亮度水平下的SNR
                    cv_temp0 = self.slid_window(sub_dict["toa"], window=window)
                    cv_temp1 = cv_temp0 * sub_dict["scale_ltypi"]  # 多波段
                    # print("window:{1}-scale_ltypi:{2}".format(window, scale_ltypi))
                    # cv_temp = cv_temp.reshape(-1)
                    if j == 0:
                        cv_win0 = cv_temp0
                        cv_win = cv_temp1
                    else:
                        cv_win0 =np.concatenate(cv_win0, cv_temp0)
                        cv_win = np.concatenate(cv_win, cv_temp1)
                # 不同窗口的cv
                cv0 = cv0 + (cv_win0,)
                cv = cv + (cv_win,)
                # CV汇总计算传感器信噪比
            hists0, snr0 = self.snr_cali(cv0)
            hists, snr = self.snr_cali(cv)
            # 到此，不同窗口下的cv全部计算出来
            columns = ["wavelength", "snr", "snr_adjust"]
            data = [self.wavelength, snr0, snr]
            output_df = pd.DataFrame(columns=columns, data=np.array(data).transpose())
            output_df['Ltoa'] = np.nanmean(toa, axis=(0, 1))
            output_df['Ltypical'] = Lty_temp
            output_df['sza']= np.nanmean(sub_dict["sza"])
            # with pd.ExcelWriter("SNR.xlsx", engine='openpyxl', mode='w') as writer:
            outfile = "SNR_revised.xlsx"
            if not os.path.exists(outfile):
                writer = pd.ExcelWriter(outfile, engine='openpyxl', mode='w')
            else:
                writer = pd.ExcelWriter(outfile, engine='openpyxl', mode='a')  # 追加
            output_df.to_excel(writer, sheet_name=os.path.basename(image_file), index=True)
            writer.close()
            # try:
            #     # 在不同版本的pandas下，写出到excel命令可能会覆盖掉本来的数据表，因此，请备份原始数据
            #     with pd.ExcelWriter("SNR.xlsx", engine='openpyxl', mode='a') as writer:
            #         output_df.to_excel(writer, sheet_name=self.sensor_id, index=True)
            # except:
            #     # 无法导出到excel，则导出到csv，通常可能是缺少openxyl库
            #     output_df.to_csv("SNR.csv", index=True)
            print(snr)
            self.drawGraph(hists, snr, self.windows, out_dir, image_file)


if __name__ == "__main__":
    SNR().main()
