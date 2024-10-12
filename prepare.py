import scipy.io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

def load_mat_file_corrected(file_path):
    mat = scipy.io.loadmat(file_path)
    data = {}
    required_vars = ['bpmECG', 'timeECG', 'sigPPG', 'sigAcc', 'sigGyro']
    for var in required_vars:
        if var in mat:
            if var == 'sigPPG':
                data[var] = mat[var].T
            elif var in ['sigAcc', 'sigGyro']:
                data[var] = (mat[var] - 32768).T
            else:
                data[var] = mat[var].flatten()
        else:
            print(f'Warning: Variable {var} not found in {file_path}.')
            data[var] = None
    return data

def baseline_correction(ppg_signal):
    return ppg_signal - np.mean(ppg_signal)

def normalize_signal(ppg_signal):
    return (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(ppg_signal, lowcut=0.5, highcut=5, fs=50, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = filtfilt(b, a, ppg_signal)
    except ValueError as e:
        print(f'Error filtering signal: {e}')
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(ppg_signal) < padlen:
            pad_width = padlen - len(ppg_signal) + 1
            ppg_signal_padded = np.pad(ppg_signal, (pad_width, pad_width), 'reflect')
            y = filtfilt(b, a, ppg_signal_padded)
            y = y[pad_width:-pad_width]
        else:
            raise e
    return y

def preprocess_ppg(ppg_signal, fs=50):
    ppg_corrected = baseline_correction(ppg_signal)
    ppg_filtered = bandpass_filter(ppg_corrected, fs=fs)
    ppg_normalized = normalize_signal(ppg_filtered)
    return ppg_normalized

def resample_signal(ppg_signal, original_fs=50, target_fs=40):
    num_samples = int(len(ppg_signal) * target_fs / original_fs)
    resampled_signal = resample(ppg_signal, num_samples)
    return resampled_signal

def synchronize_hr_labels(timeECG, bpmECG, target_fs=40, duration=None):
    if duration is None:
        duration = timeECG[-1] - timeECG[0]
    num_samples = int(duration * target_fs)
    timePPG = np.linspace(timeECG[0], timeECG[-1], num_samples)
    bpm_interpolated = np.interp(timePPG, timeECG, bpmECG)
    return bpm_interpolated, timePPG

def align_ppg_and_labels(ppg_signal, hr_labels):
    min_length = min(len(ppg_signal), len(hr_labels))
    return ppg_signal[:min_length], hr_labels[:min_length]

def sliding_window(data, window_size, step_size):
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    return windows

def plot_ppg(ppg_signal, hr_label, fs_ppg=40, title_prefix=''):
    time_ppg = np.arange(len(ppg_signal)) / fs_ppg
    plt.figure(figsize=(10, 4))
    plt.plot(time_ppg, ppg_signal, label='PPG Signal')
    plt.title(f'{title_prefix} PPG Window')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    print(f'{title_prefix} HR Label: {hr_label:.2f} BPM')

def process_subject_corrected(file_path, target_fs=40, window_duration=10, step_duration=5):
    data = load_mat_file_corrected(file_path)
    required_keys = ['bpmECG', 'timeECG', 'sigPPG']
    for key in required_keys:
        if data[key] is None:
            print(f'Skipping {file_path}: Missing {key}')
            return []
    ppg_channels = data['sigPPG']
    ppg_avg = np.mean(ppg_channels, axis=1)
    preprocessed_ppg = preprocess_ppg(ppg_avg, fs=50)
    resampled_ppg = resample_signal(preprocessed_ppg, original_fs=50, target_fs=target_fs)
    hr_labels, ppg_time = synchronize_hr_labels(
        data['timeECG'], data['bpmECG'], target_fs=target_fs, duration=data['timeECG'][-1] - data['timeECG'][0]
    )
    aligned_ppg, aligned_hr = align_ppg_and_labels(resampled_ppg, hr_labels)
    window_size = window_duration * target_fs
    step_size = step_duration * target_fs
    ppg_windows = sliding_window(aligned_ppg, window_size, step_size)
    hr_windows = sliding_window(aligned_hr, window_size, step_size)
    hr_labels_per_window = [np.mean(hr_window) for hr_window in hr_windows]
    windows = list(zip(ppg_windows, hr_labels_per_window))
    print(f'Processed {file_path}: {len(windows)} windows generated.')
    return windows

def check_signal_lengths(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.mat'):
            file_path = os.path.join(data_dir, file)
            data = load_mat_file_corrected(file_path)
            if data['sigPPG'] is not None:
                ppg_length = data['sigPPG'].shape[0]
                print(f'{file}: PPG length = {ppg_length}')
            else:
                print(f'{file}: PPG data not found.')

def is_valid_mat(file_path):
    try:
        data = load_mat_file_corrected(file_path)
        required_keys = ['bpmECG', 'timeECG', 'sigPPG', 'sigAcc', 'sigGyro']
        for key in required_keys:
            if data[key] is None:
                return False
        return True
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        return False

def visualize_some_windows(all_windows, num_samples=3):
    for i in range(min(num_samples, len(all_windows))):
        ppg_window, hr_label = all_windows[i]
        plot_ppg(ppg_window, hr_label, fs_ppg=40, title_prefix=f'Window {i+1}')

def prepare_ml_data(ppg_signal, hr_labels, test_size=0.2, val_size=0.1, random_state=42):
    X = ppg_signal  # Shape: (M, L)
    y = hr_labels   # Shape: (M,)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def main_corrected():
    data_dir = './'
    all_windows = []
    print("Checking signal lengths:")
    check_signal_lengths(data_dir)
    for file in os.listdir(data_dir):
        if file.endswith('.mat'):
            file_path = os.path.join(data_dir, file)
            if not is_valid_mat(file_path):
                print(f'Skipping {file}: Invalid or corrupted file')
                continue
            windows = process_subject_corrected(file_path, target_fs=40, window_duration=10, step_duration=5)
            if windows:
                all_windows.extend(windows)
                print(f'Processed {file}')
            else:
                print(f'Skipped {file}')
    if not all_windows:
        print('No windows were processed. Please check your data.')
        return
    X = np.array([window[0] for window in all_windows])  # Shape: (M, L)
    y = np.array([window[1] for window in all_windows])  # Shape: (M,)
    print(f'Combined dataset shape: X={X.shape}, y={y.shape}')
    visualize_some_windows(all_windows, num_samples=3)
    window_sizes = [len(window[0]) for window in all_windows]
    unique_sizes = set(window_sizes)
    if len(unique_sizes) != 1:
        print(f'Warning: Inconsistent window sizes detected: {unique_sizes}')
    else:
        print(f'All windows have a consistent size of {unique_sizes.pop()} samples.')
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_ml_data(X, y)
    print(f'Training samples: {X_train.shape[0]}, Window size: {X_train.shape[1]}')
    print(f'Validation samples: {X_val.shape[0]}, Window size: {X_val.shape[1]}')
    print(f'Testing samples: {X_test.shape[0]}, Window size: {X_test.shape[1]}')
    # Save the datasets
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    print('Data preparation complete.')

if __name__ == '__main__':
    main_corrected()
