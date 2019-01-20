import tensorflow as tf
from keras.models import load_model
from generate_data import generate
import numpy as np

def preprocess():
    signal, (label_1, label_2) = generate(1000, 6)
    input_train = np.empty((len(signal) - 20, 21))
    output_1_train = np.empty(len(signal) - 20)
    output_2_train = np.empty(len(signal) - 20)
    for i in range(len(signal) - 20):
        input_train[i] = np.array(signal[i: i + 21])
        output_1_train[i] = label_1[i]
        output_2_train[i] = label_2[i]
    return (input_train, output_1_train, output_2_train)

input_train, output_1_train, output_2_train = preprocess()
test_data = np.expand_dims(input_train, axis=2)
test_labels = output_1_train
model = load_model('model/bpsk_dm.h5')
results = model.evaluate(test_data, test_labels)
print(results)