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


file_path = 'ecg_session_data.csv'
#データの取得
header, data = read_data(file_path=file_path)
nni,arousal_level = extract_and_prepare_features(data,True)
print(nni)

plt.figure(figsize=(10, 8))
plt.plot(nni)
plt.title('NNI')
plt.tight_layout()
plt.show()