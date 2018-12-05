import numpy as np
from math import pi, ceil, floor


def add_awgn(signal, snr):
    # snr: dB
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(signal ** 2) / len(signal)
    npower = xpower / snr
    return np.random.randn(len(signal)) * np.sqrt(npower) + signal

def bpsk_module(arr):
    size = len(arr)
    sampling_arr = np.arange(0, size, 0.01)
    sampled_arr = np.zeros(len(sampling_arr), dtype=np.float32)
    for i in range(len(sampling_arr)):
        sampled_arr[i] = arr[floor(sampling_arr[i])]

    fc = 2205
    freq = 20 * fc
    coherent_carrier = np.dot(2 * pi * fc, np.arange(0, (100 * size) / freq, 1 / freq))
    bpsk = np.cos(coherent_carrier + pi * (sampled_arr - 1))
    return bpsk

def generate_label(arr):
    sampling_arr = np.arange(0, len(arr), 0.01)
    # label_arr_1 stands for phase shifting from 0 to pi
    # label_arr_2 for pi to 0
    label_arr_1 = np.zeros(len(sampling_arr), dtype=np.int8)
    label_arr_2 = np.zeros(len(sampling_arr), dtype=np.int8)
    # we don't care the last 100 points since phase shifting
    # will never happen within
    for i in range(len(sampling_arr) - 100):
        ele = sampling_arr[i]
        if ele > ceil(ele) - 0.21 and ele < ceil(ele):
            # includes 21 points before phase shifting
            data_i = int(ceil(ele))
            if arr[data_i] == 0 and arr[data_i - 1] == 1:
                # 1 -> 0, phase 0 -> pi
                label_arr_1[i] = 1
                continue
            if arr[data_i] == 1 and arr[data_i - 1] == 0:
                # 0 -> 1, phase pi -> 0
                label_arr_2[i] = 1
                continue
        label_arr_1[i] = 0
        label_arr_2[i] = 0
    return label_arr_1, label_arr_2

def generate(arr_size, snr = -2):
    # random signal size
    # ran_size = np.random.randint(16, 64)
    arr = np.random.randint(0, 2, arr_size)
    data_signal = bpsk_module(arr)
    data_signal = add_awgn(data_signal, snr)
    return (data_signal, generate_label(arr))