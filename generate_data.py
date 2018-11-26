import numpy as np
from math import pi, floor


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

def generate_label():
    arr = np.array([0, 0, 1, 0, 0, 0, 1])
    # looking for phase shifting
    phase_shift_arr = np.zeros(len(arr), dtype=np.float32)
    for i in range(1, len(arr)):
        phase_shift_arr[i] = 0 if arr[i] == arr[i-1] else 1

def main():
    # random signal size
    ran_size = np.random.randint(16, 64)
    arr = np.random.randint(0, 2, ran_size)
    # data_signal = bpsk_module(arr)
    # data_label = generate_label(arr)

    pass

# if __name__ == "__main__":
#     main()