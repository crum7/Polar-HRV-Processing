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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



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

'''
データを学習データとテストデータに分ける関数
'''
def split_data_by_id(data,arousal:bool):
    grouped_data = {}
    for line in data:
        parts = line.split(',')  # CSVファイルの場合はカンマで分割

        if arousal:
            id_arousal = f"{parts[0]}_{parts[3]}"  # Human_Num_Arousal
        else:
            id_arousal = f"{parts[0]}_{parts[2]}"  # Human_Num_Valence
        if id_arousal not in grouped_data:
            grouped_data[id_arousal] = []
        grouped_data[id_arousal].append(line)
    
    train_data = []
    test_data = []
    for id_arousal, group in grouped_data.items():
        if len(group) > 1:
            train, test = train_test_split(group, test_size=0.3, random_state=1111111)
            train_data.extend(train)
            test_data.extend(test)
        else:
            # グループのサイズが1の場合は、トレーニングデータに追加
            train_data.extend(group)

    return train_data, test_data


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


file_path = 'DREAMER_Data/CSV/emotion_nni_v2.csv'
# データをtrainデータとtestデータに分割
header, data = read_data(file_path=file_path)
nni,arousal_level = extract_and_prepare_features(data,True)
data = apply_hrv(nni)
# リストの辞書をDataFrameに変換
df = pd.DataFrame([item[0] for item in data])

X = df.drop(['mean_hr','max_hr','min_hr','std_hr'], axis=1)
y = arousal_level


# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルを作成
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# モデルの評価
score = model.score(X_test, y_test)
print("Model accuracy:", score)