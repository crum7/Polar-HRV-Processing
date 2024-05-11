#このセクションでは、データ取得と解析に必要なライブラリをインポートします。`asyncio`はPythonで非同期プログラミングを行うための標準ライブラリで、タスクの同時実行を可能にします。`bleak`はBluetooth Low Energy (BLE) デバイスとの通信を扱うためのライブラリです。`numpy`は数値計算を効率的に行うためのライブラリで、データの保存や処理に使用します。`plotly.express`はデータの可視化を行うためのライブラリで、収集したデータの視覚的な解析を容易にします。
import asyncio
from bleak import BleakScanner, BleakClient
import numpy as np
import matplotlib.pyplot as plt

# 使用するデバイスの名前
POLAR_H10_NAME = "Polar H10 B91CE12A"  

# ECGデータストリームのUUID
#変更なし
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

ecg_session_data = []
ecg_session_time = []

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

#指定されたオフセットと長さでデータを変換
def convert_array_to_signed_int(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset : offset + length]), byteorder="little", signed=True,
    )

#指定されたオフセットと長さでデータを変換
def convert_to_unsigned_long(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset : offset + length]), byteorder="little", signed=False,
    )


async def run():
    devices = await BleakScanner.discover()
    polar_h10_device = None

    # Polar H10デバイスの探索
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
        # 探索時間20秒間
        await client.connect(timeout=20.0)
        await client.write_gatt_char(PMD_CONTROL, ECG_WRITE)
        await client.start_notify(PMD_DATA, data_conv)
        # 5分間ECGデータを収集
        await asyncio.sleep(300.0)
        await client.stop_notify(PMD_DATA)
    
        # データ取得開始時に、大きなドリフトがあるので、最初の2秒のデータはカット
        ecg_cutted_session_data = ecg_session_data[:512]
        ecg_cutted_session_time = ecg_session_time[:512]

        # 収集したECGデータをファイルに保存
        np.savetxt("ecg_session_time.csv", ecg_cutted_session_time, delimiter=",")
        np.savetxt("ecg_session_data.csv", ecg_cutted_session_data, delimiter=",")
        print("ECGデータ保存")

        # ECGデータをプロット
        plt.figure(figsize=(10, 4))
        plt.plot(ecg_cutted_session_data)
        plt.title('ECG')
        plt.xlabel('Samples')
        plt.ylabel('Voltage (mV)')
        plt.tight_layout()
        plt.savefig("ecg_drift.png") 
        plt.show()


asyncio.run(run())