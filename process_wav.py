#!/usr/bin/env python
import math
import wave
import numpy as np
from scipy.fftpack import dct

def read_wav_file_into_np(filename):
    f = wave.open(filename)
    nchannels, sampwidth, framerate, nframes, _, _ = f.getparams()

    if nchannels != 1:
        print "Error: wav file is not Mono"
        return None

    if sampwidth == 1:
        dt = np.int8
    elif sampwidth == 2:
        dt = np.int16
    elif sampwidth == 4:
        dt = np.int32
    elif sampwidth == 8:
        dt = np.int64
    else:
        print "Invalid sample width:", sampwidth

    data = f.readframes(-1)
    data = np.fromstring(data, dtype=dt)
    assert(len(data) == nframes)

    f.close()
    return data, framerate


def seperate_into_windows(data, fps, window_length, gap):
    """window_length in ms, between in ms"""
    frames_per_window = int(fps*window_length/1000.)
    gap_in_frames = int(gap/1000. * fps)
    num_windows = len(data)/gap_in_frames

    windows = []
    count = 0
    for i in xrange(num_windows):
        idx = i*gap_in_frames
        
        window_data = data[idx:idx + frames_per_window]
        count += 1
        if len(window_data) < frames_per_window:
            break;

        windows.append(window_data)

    windows = np.array(windows, dtype=data.dtype)
    return windows

def get_pitch_and_energy(data, fps):
    # could multiply by hamming windows
    window = data * np.hamming(len(data))
    
    fft_data = abs(np.fft.rfft(window))**2
    fft_data_idx = fft_data[1:].argmax() + 1

    energy = np.sum(np.abs(data))/len(data)

    # quadratic interpolation around the max
    if fft_data_idx != len(fft_data) - 1:
        y0, y1, y2 = np.log(fft_data[fft_data_idx - 1:fft_data_idx + 2:])
        x1 = (y2 - y0) * 0.5 / (2 * y1 - y2 - y0)

        # find frequency
        freq = (fft_data_idx + x1) * fps/len(window)
    else:
        freq = fft_data_idx * fps/len(window)

    return freq, energy

def get_mfcc(data, fps):
    window_size = len(data)
    complex_spectrum = np.fft.fft(data)
    power_spectrum = np.abs(complex_spectrum) ** 2
    filtered_spectrum = np.dot(power_spectrum, gen_mel_filter_bank(window_size, fps))
    log_spectrum = np.log(filtered_spectrum)
    dct_spectrum = dct(log_spectrum, type=2)

    return dct_spectrum
    
def gen_mel_filter_bank(window_size, fps):
    num_coeffs = 13
    min_mel = int(freq2mel(0))
    max_mel = int(freq2mel(fps/2))
    
    filter_matrix = np.zeros((num_coeffs, window_size))

    mel_range = np.array(xrange(num_coeffs + 2))

    mel_center_filters = mel_range * (max_mel - min_mel) / (num_coeffs + 1) + min_mel
    
    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (np.exp(mel_center_filters * aux) - 1) / (fps / 2)
    aux = 0.5 + 700 * window_size * aux
    aux = np.floor(aux)
    center_index = np.array(aux, int)


    for i in xrange(num_coeffs):
        start, center, end = center_index[i:i+3]
        k1 = np.float32(center - start)
        k2 = np.float32(end - center)
        up = (np.array(xrange(start, center)) - start) / k1
        down = (end - np.array(xrange(center, end))) / k2

        filter_matrix[i][start:center] = up
        filter_matrix[i][center:end] = down

    return filter_matrix.T

def freq2mel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def mel2freq(mel):
    return 700 * (math.exp(freq / 1127.01048 - 1))

def get_diffs(data, distance=2):
    repeat = distance/2
    filter = [-1, 1]*repeat
    if distance % 2:
        filter.append(-1)

    if len(filter) > len(data):
        print "Error: length of data too short to calculate differences over that distance"

    return np.convolve(data, filter, mode='valid')

def seperate_features_into_windows(data, window_length, gap):
    """ window_length, gap is in number of frames """
    """ data is a list of features of variable length """
    windows = []
    num_windows = len(data)/gap
    for i in xrange(num_windows):
        idx = i*gap
        
        window_data = data[idx:idx + window_length]
        if len(window_data) < window_length:
            break;

        window_data = np.array(window_data)
        window_data = window_data.reshape((window_data.size), 1)
        print window_data.shape

        windows.append(window_data)

    windows = np.array(windows)
    return windows

if __name__ == "__main__":
    data, fps = read_wav_file_into_np("in.wav")
    windows = seperate_into_windows(data, fps, 25, 10)
    pitch_vectors = []
    energy_vectors = []
    mfcc_vectors = []
    for i in xrange(windows.shape[0]):
        pitch, energy = get_pitch_and_energy(windows[i, :], fps)
        mfcc = get_mfcc(windows[i, :], fps)
        pitch_vectors.append(pitch)
        energy_vectors.append(energy)
        mfcc_vectors.append(mfcc)

        print "processing:", i, "/", windows.shape[0]

    frame_length = 100
    frame_gap = 10
    pitch_features = seperate_features_into_windows(pitch_vectors, frame_length, frame_gap)
    energy_features = seperate_features_into_windows(energy_vectors, frame_length, frame_gap)
    mfcc_features = seperate_features_into_windows(mfcc_vectors, frame_length, frame_gap)

