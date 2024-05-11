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