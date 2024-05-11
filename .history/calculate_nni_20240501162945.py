file_path = 'DREAMER_Data/CSV/emotion_concat_ecg_v2_preprocessed.csv'
output_file = 'DREAMER_Data/CSV/emotion_nni_v2.csv'


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
    try:
        ecg_list = ast.literal_eval(ecg_list_str)
    except ValueError:
        ecg_list = []
    print('human',human)

    # ecg_listの各要素を個別のカラムとして扱う
    data_list.append([human, video_num, valence, arousal, dominance, age, gender] + list(ecg_list))

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
    rri_list = []
    signal = row[ecg_columns].dropna().tolist()
    print(signal)

    rri_list.append(calculate_rri(signal, 256))
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
        'rri_list': rri_list[0][0].tolist()
    }
    
    # 新しい行をファイルに追加
    with open(output_file, 'a') as f:
        f.write(','.join(map(str, new_row.values())) + '\n')
