import tensorflow as tf
import keras


def create_model(input_size):
    carrier_input = keras.layers.Input(shape=(1, input_size+1))
    first_cnn = keras.layers.Conv1D(input_size+1,
                                    padding="same",
                                    activation="relu",
                                    kernel_size=8)(carrier_input)
    second_cnn = keras.layers.Conv1D(input_size+1,
                                    padding="same",
                                    activation="relu",
                                    kernel_size=8)(carrier_input)
    merged_cnn = keras.layers.concatenate([first_cnn, second_cnn])
    merged_cnn = keras.layers.Dense(2 * input_size + 4, activation="relu")(merged_cnn)
    phase_shift = keras.layers.Dense(1, activation="relu")(merged_cnn)
    model = keras.models.Model(inputs=carrier_input, outputs=phase_shift)
    model.summary()
    
    return model