import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dropout, Input
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Reshape, TimeDistributed
from tensorflow.keras.layers import LSTM, Bidirectional, MaxPooling1D, GlobalMaxPooling1D, RepeatVector
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization  # , ConvLSTM1D




def CNN():

    inp = Input(shape=(13055,), name='x')

    x = Reshape((13055, 1), input_shape=(13055,))(inp)
    x = Conv1D(8, kernel_size=7, padding='same', strides=3, activation='relu')(x)
    x = MaxPooling1D(4, strides=2, padding='same')(x)
    x = Conv1D(64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(4, strides=2, padding='same')(x)
    x = Conv1D(64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(4, strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)

    arousal = Dense(units=2, activation='softmax', name='arousal')(x)
    valence = Dense(units=2, activation='softmax', name='valence')(x)
    model = Model(inputs=inp, outputs=[arousal, valence])

    return model

def STACKED_LSTM(arch):

    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)
    x = LSTM(10, return_sequences=True)(x)
    x = LSTM(100)(x)
    x = Reshape((100, 1), input_shape=(100,))(x)
    x = Dropout(0.2)(x)
    x = LSTM(10, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=64, activation='relu')(x)

    arousal = Dense(units=2, activation='softmax', name='arousal')(x)
    valence = Dense(units=2, activation='softmax', name='valence')(x)
    model = Model(inputs=inp, outputs=[arousal, valence])

    return model


def AUTOENCODER_LSTM(arch):

    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)

    x = LSTM(100, return_sequences=False)(x)
    x = RepeatVector(10)(x)
    x = LSTM(10, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = TimeDistributed(Dense(units=10, activation='relu'))(x)
    x = Flatten()(x)

    arousal = Dense(units=2, activation='softmax', name='arousal')(x)
    valence = Dense(units=2, activation='softmax', name='valence')(x)
    model = Model(inputs=inp, outputs=[arousal, valence])

    return model


def BI_LSTM():

    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)
    x = Bidirectional(LSTM(1000, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(10, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)

    arousal = Dense(units=2, activation='softmax', name='arousal')(x)
    valence = Dense(units=2, activation='softmax', name='valence')(x)
    model = Model(inputs=inp, outputs=[arousal, valence])

    return model



class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'arousal_loss': [], 'valence_loss': [],
                        'arousal_accuracy': [], 'valence_accuracy': []
                        }

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

        self.history['arousal_loss'].append(logs.get('arousal_loss'))
        self.history['valence_loss'].append(logs.get('valence_loss'))

        self.history['arousal_accuracy'].append(logs.get('arousal_accuracy'))
        self.history['valence_accuracy'].append(logs.get('valence_accuracy'))

