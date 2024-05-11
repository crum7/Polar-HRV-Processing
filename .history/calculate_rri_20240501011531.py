import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import japanize_matplotlib
from datetime import datetime,timedelta
from scipy.signal import resample
import matplotlib.dates as mdates
import itertools
import numpy as np
from scipy import interpolate
import statistics
from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
import scipy.signal
import scipy.interpolate
from biosppy.signals import ecg
import pyhrv.time_domain as td
import csv
import sys
sys.path.append('./')  # カレントディレクトリをパスに追加
from sklearn.preprocessing import LabelEncoder
import ast
'''
emotion_concat_ecg.csvから、rriを求める
'''

def get_rri2(signal, sampling_rate):
    # R波のピークの初期検出
    out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
    initial_rpeaks = out['rpeaks']

    # R波のピークの補正
    rpeaks_hamilton = ecg.correct_rpeaks(signal=signal, rpeaks=initial_rpeaks, sampling_rate=sampling_rate)

    # RRIの計算（ミリ秒単位）
    rri = np.diff(rpeaks_hamilton) * 1000 / sampling_rate
    print(rri)
    return rri





file_path = 'DREAMER_dataset/preprocess/table_ecg_concat_v2.csv'
output_file = 'DREAMER_dataset/preprocess/emotion_rri_v2.csv'


with open(file_path, 'r') as file:
    lines = file.readlines()

header = lines[0].strip().split(',')

data_list = []
for line in lines[1:]:  # skip header
    split_line = line.strip().split(',')
    
    # メタデータの抽出
    human = int(split_line[0])  # Human_Num
    video_num = int(split_line[1])  # Video_Num
    valence = float(split_line[2])  # Valence
    arousal = float(split_line[3])  # Arousal
    dominance = float(split_line[4])  # ScoreDominance
    age = int(split_line[5])  # Age
    gender = int(split_line[6])  # Gender
    
    # ecg_listの設定
    ecg_list_str = ','.join(split_line[7:])  # ecg_list
    try:
        ecg_list = ast.literal_eval(ecg_list_str)
    except ValueError:
        ecg_list = []
    print('human',human)

    # ecg_listの各要素を個別のカラムとして扱う
    data_list.append([human, video_num, valence, arousal, dominance, age, gender] + list(ecg_list))



#次は、train_test_splitでtensor_ecg_listを分ける
# 最大のecg_listの長さを取得して、カラム名を動的に作成
max_ecg_length = max(len(item[7:]) for item in data_list)
ecg_columns = [f"ecg_{i}" for i in range(max_ecg_length)]
#2.データの読み込み
df = pd.DataFrame(data_list, columns=header[:7] + ecg_columns)

# 最初の行のためにヘッダーを書き込む
with open(output_file, 'w') as f:
    f.write(','.join([
        'Human_Num', 'Video_Num', 'Valence',
        'Arousal', 'ScoreDominance', 'Age',
        'Gender', 'rri_list'
    ]) + '\n')

for i, row in df.iterrows():
    # ループの残りの部分はそのまま

    rri_list = []
    signal = row[ecg_columns].dropna().tolist()  # 現在の行からecgデータを取得
    print(signal)

    '''
            if signal.size == 0:
        print(f"Skipping invalid signal for column {col} at index {i}.")
        continue
    '''


    rri_list.append(get_rri2(signal, 256))
    print(rri_list)
    
    new_row = {
        'Human_Num': df.loc[i, 'Human_Num'],
        'Video_Num': df.loc[i, 'Video_Num'],
        'Valence': df.loc[i, 'Valence'],
        'Arousal': df.loc[i, 'Arousal'],
        'ScoreDominance': df.loc[i, 'ScoreDominance'],
        'Age': df.loc[i, 'Age'],
        'Gender': df.loc[i, 'Gender'],
        #numpyをlistに変換する
        'rri_list': rri_list[0].tolist()
    }
    
    # 新しい行をファイルに追加
    with open(output_file, 'a') as f:
        f.write(','.join(map(str, new_row.values())) + '\n')
