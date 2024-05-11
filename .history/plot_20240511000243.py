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
data = np.loadtxt('ecg_session_data_excite.csv')
plt.figure(figsize=(10, 8))
plt.plot(data[512:1024])
plt.title('ECG')
plt.tight_layout()
plt.show()