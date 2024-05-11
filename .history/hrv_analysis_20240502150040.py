import hrvanalysis

file_path = 'DREAMER_Data/CSV/emotion_nni_v2.csv'
#データの取得
header, data = read_data(file_path=file_path)
nni,arousal_level = extract_and_prepare_features(data,True)
print(nni[0])

# hrvanalysis.extract_features.get_time_domain_features(nn_intervals=)