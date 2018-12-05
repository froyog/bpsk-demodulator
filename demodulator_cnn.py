import tensorflow as tf
from keras.models import Model
from keras.layers import Conv1D, concatenate, Dense, Input, Flatten

def create_model(input_size):
    carrier_input = Input(shape=(21, 1))
    first_cnn = Conv1D(1,
                       padding='same',
                       activation='relu',
                       kernel_size=input_size)(carrier_input)
    second_cnn = Conv1D(1,
                        padding='same',
                        activation='relu',
                        kernel_size=input_size)(carrier_input)
    merged_cnn = concatenate([first_cnn, second_cnn])
    flat = Flatten()(merged_cnn)
    hidden = Dense(2 * input_size, activation='relu')(flat)
    phase_shift = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=carrier_input, outputs=phase_shift)
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model