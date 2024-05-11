from hrvanalysis import preprocessing

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
    print(raw_rri)
    
    cleaned_rri = preprocessing.remove_outliers(rri) 
    preprocessed_rri_list.append(raw_rri)


plt.figure(figsize=(10, 8))
plt.plot(preprocessed_rri_list)
plt.title('RRI')
plt.legend()
plt.tight_layout()
plt.show(block=False)