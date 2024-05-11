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