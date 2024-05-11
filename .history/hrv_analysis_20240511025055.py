import hrvanalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import seaborn as sns
import csv
import imageio
from scipy.signal import butter, filtfilt
import pywt
import ast
from biosppy.signals import ecg
from scipy.signal import iirnotch, lfilter
from sklearn.model_selection import train_test_split
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values




import hrvanalysis

#心拍変動解析の適用
def apply_hrv(nni_list):
    hrv_data_list = []
    
    for nni in nni_list:
        time_domain_data = hrvanalysis.extract_features.get_time_domain_features(nn_intervals=nni)
        #補間を線形で、再サンプリングを4Hzで、周波数領域を計算
        frequency_domain_data = hrvanalysis.extract_features.get_frequency_domain_features(nn_intervals=nni,method= 'welch', sampling_frequency = 4, interpolation_method= 'linear')

        hrv_data = dict(time_domain_data, **frequency_domain_data)
        hrv_data_list.append([hrv_data])

    return hrv_data_list



with open('CSV/nni_calm.txt', 'r') as f:
    data = f.read().strip('[]').split()

nni = np.array([float(x) for x in data])

#PSDのplot
hrvanalysis.plot.plot_psd(nni)
nni_list = nni.tolist()
nni_hrv = apply_hrv(nni_list)



print(nni_hrv)

    