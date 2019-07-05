from numpy import load
import matplotlib.pyplot as plt
import os

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    data = load(wav_file)
    nfft = 200 # Length of each window segment
    fs = 200 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx
