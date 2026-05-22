import mne
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


data = sio.loadmat("NEW_PY/Arithmetic_sub_1_trial1_cleaned.mat")
eeg = data["Clean_data"]

ch_names = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 
    'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 
    'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 
    'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
]

info = mne.create_info(ch_names=ch_names, sfreq=128)
raw = mne.io.RawArray(eeg, info)

raw.plot()
plt.show()