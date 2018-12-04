import numpy as np
from math import pi, floor


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
    bpsk = np.cos(coherent_carrier + pi * (sampled_arr - 1) + pi / 4)
    return bpsk

def generate_label(arr):
    # looking for phase shifting
    phase_shift_arr = np.zeros(len(arr))
    for i in range(0, len(arr) - 1):
        phase_shift_arr[i] = 0 if arr[i] == arr[i+1] else 1
    sampling_arr = np.arange(0, len(arr), 0.01)
    label_arr = np.zeros(len(sampling_arr))
    for j in range(len(sampling_arr)):
        ele = sampling_arr[j]
        if ele > round(ele) - 0.2 and ele < round(ele):
            # include all 20 points before phase shifting
            label_arr[j] = phase_shift_arr[int(round(ele)) - 1]
            continue
        label_arr[j] = 0
    return label_arr

def main():
    # random signal size
    ran_size = np.random.randint(16, 64)
    arr = np.random.randint(0, 2, ran_size)
    data_signal = bpsk_module(arr)
    data_signal = add_awgn(data_signal, 2)
    data_label = generate_label(arr)

main()
# if __name__ == "__main__":
#     main()