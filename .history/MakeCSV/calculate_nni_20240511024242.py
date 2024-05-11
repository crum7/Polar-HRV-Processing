from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
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
# import heartpy
from biosppy.signals import ecg
from scipy.signal import iirnotch, lfilter
from sklearn.model_selection import train_test_split
# from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
'''
RR間隔(RRI)の計算
'''

def calculate_rri(signal, sampling_rate):
    # R波のピークの初期検出
    out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
    initial_rpeaks = out['rpeaks']

    # R波のピークの補正
    rpeaks_hamilton = ecg.correct_rpeaks(signal=signal, rpeaks=initial_rpeaks, sampling_rate=sampling_rate)

    # RRIの計算（ミリ秒単位）
    rri = np.diff(rpeaks_hamilton) * 1000 / sampling_rate
    return rri


def calculate_nni(raw_rri):
    # 生のRR間隔データから外れ値(outlier)を除去する
    cleaned_rri = remove_outliers(rr_intervals=raw_rri, verbose=True)

    # 外れ値を除去したデータのNaN(欠損値)を線形補間で穴埋めする
    fill_nan_rri = interpolate_nan_values(rr_intervals=cleaned_rri, interpolation_method="linear")

    # 補間済みRR間隔データから異所性心拍(ectopic beat)を除去し、NN間隔を得る
    nn_intervals_list = remove_ectopic_beats(rr_intervals=fill_nan_rri, method="malik")

    # NN間隔のNaN(欠損値)を線形補間で穴埋めする 
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    return interpolated_nn_intervals

#新しいファイルへの書き込み 
new_file_path = 'CSV/nni_calm.txt'
rri_list = []

#Polar H10のサンプリング周波数
fs = 130
# ECGの取得
file_path = 'CSV/ecg_calm.txt'  
ecg_signal = np.loadtxt(file_path)
# RRIの計算
rri_list = calculate_rri(ecg_signal,fs)
rri_list = rri_list
rri_list = rri_list[1:-1]
print(rri_list)
nni_list = calculate_nni(rri_list)


# 前処理を全て適用済
with open(new_file_path, 'w') as f:
    f.write(np.array2string(rri_list[0], precision=18, separator=' ', suppress_small=True))