#このセクションでは、データ取得と解析に必要なライブラリをインポートします。`asyncio`はPythonで非同期プログラミングを行うための標準ライブラリで、タスクの同時実行を可能にします。`bleak`はBluetooth Low Energy (BLE) デバイスとの通信を扱うためのライブラリです。`numpy`は数値計算を効率的に行うためのライブラリで、データの保存や処理に使用します。`plotly.express`はデータの可視化を行うためのライブラリで、収集したデータの視覚的な解析を容易にします。
import asyncio
from bleak import BleakScanner, BleakClient
import numpy as np
import matplotlib.pyplot as plt

#このセルでは、接続するデバイスの名前やBluetoothの特性を指定するUUIDなど、プログラム全体で使用する定数を設定します。また、セッション中に収集されるECGデータとタイムスタンプを保存するためのリストを初期化します。これにより、データ取得中に各心拍の値とその受信時刻を記録できるようになります。

# 使用するデバイスの名前
POLAR_H10_NAME = "Polar H10 B91CE12A"  

# ECGデータストリームのUUID
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

ecg_session_data = []  # ECGセッションのデータを保存するリスト
ecg_session_time = []  # ECGセッションのタイムスタンプを保存するリスト
#データ受信時に実行されるコールバック関数`data_conv`を定義します。この関数は、Bluetooth経由で受信した生データを解析し、心拍のサンプル値とタイムスタンプを抽出してグローバルリストに保存します。また、データを符号付き整数または符号なし長整数に変換するヘルパー関数も含まれています。これにより、受信したバイト列から数値データを正しく抽出し、分析可能な形式に変換します。

#受信したデータの処理
def data_conv(sender, data):
    print("Data received")
    if data[0] == 0x00:
        timestamp = convert_to_unsigned_long(data, 1, 8)
        step = 3
        samples = data[10:]
        offset = 0
        while offset < len(samples):
            ecg = convert_array_to_signed_int(samples, offset, step)
            offset += step
            ecg_session_data.extend([ecg])
            ecg_session_time.extend([timestamp])
            
        #polar H10はデータ取得開始時に、大きなドリフトがあるので、最初の2秒のデータはカットする
        ecg_session_data[:512]
        ecg_session_time[:512]
        

#指定されたオフセットと長さでデータを符号付き整数に変換
def convert_array_to_signed_int(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset : offset + length]), byteorder="little", signed=True,
    )

#指定されたオフセットと長さでデータを符号なしの長整数に変換する関数
def convert_to_unsigned_long(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset : offset + length]), byteorder="little", signed=False,
    )

#メインの非同期関数`run`では、デバイスのスキャン、接続、データ取得の開始、データの保存、そして可視化までのプロセスが定義されています。この関数はデバイスの検出からデータの可視化までを一連のステップとして実行し、各ステップで進捗状況や結果をコンソールに出力します。非同期処理により、UIのフリーズを防ぎながら効率的にデータ収集と処理を行います。

#メインの非同期関数
async def run():
    devices = await BleakScanner.discover()
    polar_h10_device = None

    # Polar H10デバイスを探す
    for device in devices:
        if device.name and POLAR_H10_NAME in device.name:
            print(f"Polar H10を見つけました: {device}")
            polar_h10_device = device
            break

    if not polar_h10_device:
        print("Polar H10が見つかりませんでした!")
        return

    # Polar H10デバイスとの接続とデータ取得
    async with BleakClient(polar_h10_device) as client:
        await client.connect(timeout=20.0)
        await client.write_gatt_char(PMD_CONTROL, ECG_WRITE)
        await client.start_notify(PMD_DATA, data_conv)
        await asyncio.sleep(10.0)  # 10秒間ECGデータを収集
        await client.stop_notify(PMD_DATA)

        # 収集したECGデータをファイルに保存
        np.savetxt("ecg_session_time.csv", ecg_session_time, delimiter=",")
        np.savetxt("ecg_session_data.csv", ecg_session_data, delimiter=",")
        print("ECGデータ保存")

        # ECGデータをプロット
        plt.figure(figsize=(10, 4))  # グラフのサイズを設定
        plt.plot(ecg_session_data[512:], label='ECG Data')  # データとラベルをプロット
        plt.xlabel('Sample Number')  # x軸のラベル
        plt.ylabel('ECG Amplitude')  # y軸のラベル
        plt.title('ECG Session Data')  # グラフのタイトル
        plt.grid(True)  # グリッドを表示
        plt.legend()  # 凡例を表示
        plt.show()  # グラフを表示

asyncio.run(run())