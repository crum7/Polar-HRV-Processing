import hrvanalysis
import numpy as np

# 心拍変動解析の適用
def apply_hrv(nni_list):
    hrv_data_list = []
    nni = np.array(nni_list)  # nni_listをNumPy配列に変換
    time_domain_data = hrvanalysis.extract_features.get_time_domain_features(nn_intervals=nni)
    frequency_domain_data = hrvanalysis.extract_features.get_frequency_domain_features(nn_intervals=nni, method='welch', sampling_frequency=4, interpolation_method='linear')
    hrv_data = dict(time_domain_data, **frequency_domain_data)
    hrv_data_list.append([hrv_data])
    return hrv_data_list

with open('CSV/nni_calm.txt', 'r') as f:
    data = f.read().strip('[]').split()
    nni_list = [float(x) for x in data]

# PSDのplot
nni = np.array(nni_list)
hrvanalysis.plot.plot_psd(nni)

nni_hrv = apply_hrv(nni_list)
print(nni_hrv)



import heartpy as hp

def analyze_hr_variability(rri):
    rri = np.array(rri)
    if rri.size < 2:
        return "RRIデータが不足しています。"

    # RRIデータを心拍数データに変換
    hr_data = 60000 / rri

    # RRIデータの差分を取り、サンプリングレートを計算
    rri_diff = np.diff(rri)
    if rri_diff.size == 0 or np.all(rri_diff == 0):
        return "RRIデータの差分が計算できません。データが一定か、不足しています。"

    sampling_rate = 256

    # 心拍変動データの解析
    try:
        wd, m = hp.process(hr_data, sample_rate=sampling_rate)
    except Exception as e:
        return f"解析中にエラーが発生しました: {str(e)}"

    metrics = {
        'SDNN': m.get('sdnn', 'データ不足'),
        'RMSSD': m.get('rmssd', 'データ不足'),
        'pNN50': m.get('pnn50', 'データ不足'),
        'LF': m.get('lf', 'データ不足'),
        'HF': m.get('hf', 'データ不足'),
        'LF/HF ratio': m.get('lf/hf', 'データ不足'),
        'Total Power': m.get('total_power', 'データ不足'),
        'LFnu': m.get('lf_nu', 'データ不足'),
        'HFnu': m.get('hf_nu', 'データ不足'),
        'CCVTP': np.std(rri) / np.mean(rri)
    }

    return metrics