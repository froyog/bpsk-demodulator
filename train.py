import tensorflow as tf
import keras
import numpy as np
import json
import matplotlib.pyplot as plt
from generate_data import generate
from demodulator_cnn import create_model

def preprocess():
    signal, (label_1, label_2) = generate()
    input_train = np.empty((len(signal) - 20, 21))
    output_1_train = np.empty(len(signal) - 20)
    output_2_train = np.empty(len(signal) - 20)
    for i in range(len(signal) - 20):
        input_train[i] = np.array(signal[i: i + 21])
        output_1_train[i] = label_1[i]
        output_2_train[i] = label_2[i]
    return (input_train, output_1_train, output_2_train)

def train():
    input_train, output_1_train, output_2_train = preprocess()
    model_1 = create_model(21)
    input_val = input_train[:10000]
    input_val = np.expand_dims(input_val, axis=2)
    partial_input_train = input_train[10000:]
    partial_input_train = np.expand_dims(partial_input_train, axis=2)
    output_1_val = output_1_train[:10000]
    partial_output_1_train = output_1_train[10000:]
    history = model_1.fit(partial_input_train,
                          partial_output_1_train,
                          epochs=100,
                          batch_size=512,
                          validation_data=(input_val, output_1_val),
                          verbose=1)
    model_1.save('model/bpsk_dm.h5')
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
    # model_2 = create_model()


if __name__ == '__main__':
    train()