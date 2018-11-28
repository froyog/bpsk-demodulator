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
    arr = np.array([0, 1, 0, 1])
    # looking for phase shifting
    phase_shift_arr = np.zeros(len(arr))
    for i in range(1, len(arr)):
        phase_shift_arr[i] = 0 if arr[i] == arr[i-1] else 1
    print(phase_shift_arr)
    sampling_arr = np.arange(0, len(arr), 0.01)
    label_arr = np.zeros(len(sampling_arr))
    for j in range(len(sampling_arr)):
        ele = sampling_arr[j]
        if ele < 1:
            label_arr[j] = 0
            continue
        if ele > round(ele) - 0.2 and ele < round(ele) + 0.2:
            label_arr[j] = phase_shift_arr[int(round(ele)) - 1]
            continue
        label_arr[j] = 0
    print(label_arr[0:100])
    print(label_arr[100:200])
    print(label_arr[200:300])
    print(label_arr[300:400])


generate_label()

def main():
    # random signal size
    ran_size = np.random.randint(16, 64)
    arr = np.random.randint(0, 2, ran_size)
    # data_signal = bpsk_module(arr)
    # data_label = generate_label(arr)

    pass

# if __name__ == "__main__":
#     main()