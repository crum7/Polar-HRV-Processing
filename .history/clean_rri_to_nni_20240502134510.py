from hrvanalysis import preprocessing
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



'''
ファイルからデータを読み込む関数
'''
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header = lines[0].strip()  # ヘッダー行を保持
    data = [line.strip() for line in lines[1:]]  # データ行を読み込み
    return header, data
'''
RRIの抽出とラベルに分ける
'''
def extract_and_prepare_features(data, arousal: bool):
    X = []
    Y = []

    for line in data[1:]:  # ヘッダーをスキップ
        split_line = line.strip().split(',')

        # RRIデータと関連する特徴量を取得
        age = int(split_line[5])
        gender = int(split_line[6])
        rri_list_str = ','.join(split_line[7:])
        try:
            rri = ast.literal_eval(rri_list_str)
            if len(rri) >= 200:
                # 中央から200個のRRIデータを取得
                mid_index = len(rri) // 2
                start_index = max(0, mid_index - 100)
                end_index = start_index + 200
                central_rri = rri[start_index:end_index]

                # RRIデータに年齢と性別を追加
                extended_data = central_rri + [age, gender]

                X.append(extended_data)
                if arousal:
                    Y.append(float(split_line[3]))  # arousalをYとして追加
                else:
                    Y.append(float(split_line[2]))  # arousalをYとして追加
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing RRI data on line {line}: {e}")

    X = np.array(X)
    Y = np.array(Y) - 1
    return X, Y



file_path = 'DREAMER_Data/CSV/emotion_rri_v2.csv'
#データの取得
header, data = read_data(file_path=file_path)
rri_list,arousal_level = extract_and_prepare_features(data,True)
#期外収縮の修正
lower_bound = 600
upper_bound = 1000

preprocessed_rri_list = []
for rri in rri_list:
    raw_rri = np.array(rri)
    # 生のRR間隔データから外れ値(outlier)を除去する
    cleaned_rri = remove_outliers(rr_intervals=raw_rri, verbose=True)

    # 外れ値を除去したデータのNaN(欠損値)を線形補間で穴埋めする
    fill_nan_rri = interpolate_nan_values(rr_intervals=cleaned_rri, interpolation_method="linear")

    # 補間済みRR間隔データから異所性心拍(ectopic beat)を除去し、NN間隔を得る
    nn_intervals_list = remove_ectopic_beats(rr_intervals=fill_nan_rri, method="malik")

    # NN間隔のNaN(欠損値)を線形補間で穴埋めする 
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    preprocessed_rri_list.append(interpolated_nn_intervals)




plt.figure(figsize=(10, 8))
plt.plot(preprocessed_rri_list)
plt.title('NNI')
plt.show()



