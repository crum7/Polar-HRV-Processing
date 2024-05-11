import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import ast


'''
時定数0.03秒のフィルタをかける
時定数が0.03秒の時定数は、5.3Hz 以下(周期0. 2秒以上)の成分は除去され低周波成分、例えばドリフトなど）の振幅を1/2に減衰することができる。基線の変動や低周波のノイズが抑制されるため、心電図（ECG）の信号がよりクリアになる。
'''


# 心電図データのサンプリング周波数（Hz）
fs = 256

# フィルタ設計用の関数
def design_highpass_filter(fs, cutoff_frequency):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return b, a

# 時定数からカットオフ周波数への変換
def time_constant_to_cutoff_frequency(time_constant):
    return 1 / (2 * np.pi * time_constant)

# 時定数0.03秒と1秒のカットオフ周波数
cutoff_freq_0_03 = time_constant_to_cutoff_frequency(0.03)
print('cutoff_freq_0_03',cutoff_freq_0_03)
cutoff_freq_1 = time_constant_to_cutoff_frequency(1)

# フィルタの設計
b_0_03, a_0_03 = design_highpass_filter(fs, cutoff_freq_0_03)
b_1, a_1 = design_highpass_filter(fs, cutoff_freq_1)

# ECGの取得
file_path = 'DREAMER_dataset/preprocess/OutputCSV/table_ecg_concat_v2.csv'
with open(file_path, 'r') as file:
    lines = file.readlines()
header = lines[0].strip().split(',')

for line in lines[1:]:  # skip header
    split_line = line.strip().split(',')

    
    # ecgデータと関連する特徴量を取得
    age = int(split_line[5])
    gender = int(split_line[6])
    ecg_list_str = ','.join(split_line[7:])
    
    try:
        ecg = ast.literal_eval(ecg_list_str)
        # 中央の15,360個(60秒間)のECGを抽出
        list_length = len(ecg)
        mid_index = list_length // 2
        start_index = mid_index - 7680
        end_index = mid_index + 7680
        ecg_cutted = ecg[start_index:end_index]
        ecg_signal = ecg_cutted

        # フィルタリング
        filtered_ecg_0_03 = filtfilt(b_0_03, a_0_03, ecg_signal)
        filtered_ecg_1 = filtfilt(b_1, a_1, ecg_signal)

        # 結果のプロット
        plt.figure(figsize=(10, 8))
        plt.subplot(311)
        plt.plot(ecg_signal, label='Original ECG')
        plt.title('Original ECG Signal')
        plt.legend()

        plt.subplot(312)
        plt.plot(filtered_ecg_0_03, label='Filtered ECG (τ=0.03s)')
        plt.title('Filtered ECG with τ=0.03s')
        plt.legend()

        plt.subplot(313)
        plt.plot(filtered_ecg_1, label='Filtered ECG (τ=1s)')
        plt.title('Filtered ECG with τ=1s')
        plt.legend()

        plt.tight_layout()
        plt.show()


    except (ValueError, SyntaxError) as e:
        print(f"Error parsing ecg data on line {line}: {e}")


