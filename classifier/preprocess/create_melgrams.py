import os
import sys

import librosa
import numpy as np
import pandas as pd


datafolder = "data"
index_path = "data/index/gtzan_genre.csv"
dst = datafolder + "/melgrams.pkl"

# mel-spectrogram parameters
SR = 12000
N_FFT = 512
N_MELS = 96
HOP_LEN = 256
DURA = 29.12  # to make it 1366 frame..


i=0
def compute_melgram(audio_path):
    global i
    sys.stdout.write("\rComputing melgrams: %i" % i)
    sys.stdout.flush()
    i+=1

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


if __name__ == "__main__":

    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    index = pd.read_csv(index_path, index_col=[0])

    index['melgram'] = index['filepath'].apply(compute_melgram)

    index.to_pickle(dst)

    print()



