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
plt.plot(data[11000:14000])
plt.title('ECG')
# Label the x-axis and y-axis
plt.xlabel('Samples')  # Update this label as appropriate for your data
plt.ylabel('Voltage (mV)')     # Update this label as appropriate for your data

plt.tight_layout()
plt.savefig("ecg_drift.png") 
plt.show()