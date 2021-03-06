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
    
    fft_ret = np.fft.rfft(window)
    fft_data = abs(fft_ret)**2
    fft_data_idx = fft_data[1:].argmax() + 1

    energy = np.sum(np.abs(data))/len(data)

    # quadratic interpolation around the max
    if fft_data_idx != len(fft_data) - 1:
        y0, y1, y2 = np.log(fft_data[fft_data_idx - 1:fft_data_idx + 2:])
        denom = (2 * y1 - y2 - y0)
        if math.isnan(denom):
            freq = 0
        elif denom == 0:
            freq = fft_data_idx * fps/len(window)
        else:
            x1 = (y2 - y0) * 0.5 / denom

            # find frequency
            freq = (fft_data_idx + x1) * fps/len(window)
    else:
        freq = fft_data_idx * fps/len(window)

    return freq, energy, fft_data

def get_mfcc(data, fps):
    window_size = len(data)
    complex_spectrum = np.fft.fft(data)
    power_spectrum = np.abs(complex_spectrum) ** 2
    filtered_spectrum = np.dot(power_spectrum, gen_mel_filter_bank(window_size, fps))
    log_spectrum = np.log(filtered_spectrum)
    log_spectrum[np.where(np.isinf(log_spectrum))] = 0
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

    return np.convolve(data, filter, mode='same')

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

        windows.append(window_data)

    windows = np.array(windows)
    windows = np.squeeze(windows)
    return windows

def concatenate_features(features):
    return np.concatenate((features), axis=1)

def gen_features_from_wave_file(filename):
    data, fps = read_wav_file_into_np(filename)
    windows = seperate_into_windows(data, fps, 25, 10)
    pitch_vectors = []
    energy_vectors = []
    mfcc_vectors = []
    fft_vectors = []
    for i in xrange(windows.shape[0]):
        pitch, energy, fft_data = get_pitch_and_energy(windows[i, :], fps)

        mfcc = get_mfcc(windows[i, :], fps)
        pitch_vectors.append(pitch)
        energy_vectors.append(energy)
        mfcc_vectors.append(mfcc)
        fft_vectors.append(fft_data)

        #print "processing:", i, "/", windows.shape[0] - 1

    frame_length = 100
    frame_gap = 10
    
    energy_features = seperate_features_into_windows(energy_vectors, frame_length, frame_gap)
    pitch_features = seperate_features_into_windows(pitch_vectors, frame_length, frame_gap)
    mfcc_features = seperate_features_into_windows(mfcc_vectors, frame_length, frame_gap)
    fft_features = seperate_features_into_windows(fft_vectors, frame_length, frame_gap)
    
    pitch_dt_features = get_diffs(pitch_vectors)
    pitch_dt_features = seperate_features_into_windows(pitch_dt_features, frame_length, frame_gap)
    
    energy_dt_features = get_diffs(energy_vectors)
    energy_dt_features = seperate_features_into_windows(energy_dt_features, frame_length, frame_gap)
    energy_dt2_features = get_diffs(energy_vectors, 3)
    energy_dt2_features = seperate_features_into_windows(energy_dt2_features, frame_length, frame_gap)

    # all_features = concatenate_features((pitch_features, energy_features, pitch_dt_features, energy_dt_features, mfcc_features))
    # all_features = concatenate_features((pitch_dt_features, energy_dt_features))
    # all_features = concatenate_features((energy_dt_features, energy_dt2_features))
    # all_features = fft_features
    all_features = concatenate_features((pitch_features, energy_features, pitch_dt_features, energy_dt_features, mfcc_features))
    print all_features.shape
    return all_features

if __name__ == "__main__":
    all_features = gen_features_from_wave_file("in.wav")
