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


file_path = 'DREAMER_Data/CSV/emotion_concat_ecg_v2_preprocessed.csv'
output_file = 'DREAMER_Data/CSV/emotion_nni_v2.csv'
# 最初の行のためにヘッダーを書き込む
with open(output_file, 'w') as f:
    f.write(','.join([
        'Human_Num', 'Video_Num', 'Valence',
        'Arousal', 'ScoreDominance', 'Age',
        'Gender', 'NNI'
    ]) + '\n')


def get_rri(signal, sampling_rate):
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


# 時定数からカットオフ周波数への変換
def time_constant_to_cutoff_frequency(time_constant):
    return 1 / (2 * np.pi * time_constant)

with open(file_path, 'r') as file:
    lines = file.readlines()

header = lines[0].strip().split(',')

data_list = []
for line in lines[1:]:  # skip header
    split_line = line.strip().split(',')
    
    # メタデータの抽出
    human = int(split_line[0])
    video_num = int(split_line[1])
    valence = float(split_line[2])
    arousal = float(split_line[3])
    dominance = float(split_line[4])
    age = int(split_line[5])
    gender = int(split_line[6])

    ecg_list_str = ','.join(split_line[7:])
    ecg_list = ast.literal_eval(ecg_list_str)
    rri_list = []
    signal = ecg_list

    #rri
    raw_rri = get_rri(signal, 256)
    #nni
    nni = calculate_nni(raw_rri[0])
    
    new_row = {
        'Human_Num': human,
        'Video_Num': video_num,
        'Valence': valence,
        'Arousal': arousal,
        'ScoreDominance': dominance,
        'Age': age,
        'Gender': gender,
        'NNI': nni
    }
    
    # 新しい行をファイルに追加
    with open(output_file, 'a') as f:
        f.write(','.join(map(str, new_row.values())) + '\n')
