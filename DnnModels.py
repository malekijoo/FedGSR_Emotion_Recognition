import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dropout, Input
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Reshape, TimeDistributed
from tensorflow.keras.layers import LSTM, Bidirectional, MaxPooling1D, GlobalMaxPooling1D, RepeatVector
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization  # , ConvLSTM1D



#
#
#
# def CNN():
#
#     model = Sequential()
#     model.add(Reshape((13055, 1), input_shape=(13055,)))
#     model.add(Conv1D(8, kernel_size=7, padding='same', strides=3, activation='relu'))
#     model.add(MaxPooling1D(4, strides=2, padding='same'))
#     # model.add(Conv1D(128, kernel_size=7, padding='same', strides=3, activation='relu'))
#     # model.add(MaxPooling1D(4, strides=2, padding='same'))
#     model.add(Conv1D(16, kernel_size=3, padding='same', strides=1, activation='relu'))
#     model.add(MaxPooling1D(4, strides=2, padding='same'))
#     model.add(Conv1D(16, kernel_size=3, padding='same', strides=1, activation='relu'))
#     model.add(MaxPooling1D(4, strides=2, padding='same'))
#     model.add(Flatten())
#     model.add(Dense(units=128, activation='relu'))
#     model.add(Dense(units=64, activation='relu'))
#     model.add(Dense(units=2, activation='softmax'))
#
#     return model
#


def CNN(arch):
    if arch == 'FED':
        unit = 1
        activ = 'sigmoid'
    elif arch == 'CENT':
        unit = 2
        activ = 'softmax'

    # model = Sequential()
    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)
    x = Conv1D(8, kernel_size=7, padding='same', strides=3, activation='relu')(x)
    x = MaxPooling1D(4, strides=2, padding='same')(x)
    # x = Conv1D(64, kernel_size=7, padding='same', strides=3, activation='relu')(x)
    # x = MaxPooling1D(4, strides=2, padding='same')(x)
    x = Conv1D(64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(4, strides=2, padding='same')(x)
    x = Conv1D(64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling1D(4, strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)

    arousal = Dense(units=unit, activation=activ, name='arousal')(x)
    valence = Dense(units=unit, activation=activ, name='valence')(x)

    model = Model(inputs=inp, outputs={'arousal': arousal, 'valence': valence})

    return model


# def conv_LSTM(self):
#     model = Sequential()
#     model.add(ConvLSTM1D(32, kernel_size=5, padding="same", return_sequences=True, activation="relu"))
#     model.add()
#     model.add()
#     model.add()
#     model.add()
#     model.add()
#     model.add()
#
#     return model
#
#     l11 = ConvLSTM1D(32, kernel_size=5, padding="same",
#                      return_sequences=True, activation="relu")(tf.expand_dims(self._input[0], axis=-1))
#     l11 = BatchNormalization()(l11)
#
#     l12 = ConvLSTM2D(32, kernel_size=(5, 5), padding="same",
#                      return_sequences=True, activation="relu")(tf.expand_dims(self._input[1], axis=-1))
#     l12 = BatchNormalization()(l12)
#
#     l21 = ConvLSTM1D(32, kernel_size=3, padding="same", return_sequences=True, activation="relu")(l11)
#     l21 = BatchNormalization()(l21)
#
#     l22 = ConvLSTM2D(32, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(l12)
#     l22 = BatchNormalization()(l22)
#
#     l31 = ConvLSTM1D(32, kernel_size=1, padding="same", return_sequences=True, activation="relu")(l21)
#     l32 = ConvLSTM2D(32, kernel_size=(1, 1), padding="same", return_sequences=True, activation="relu")(l22)
#
#     l31 = Dropout(0.2)(l31)
#     l32 = Dropout(0.2)(l32)
#
#     l31 = Flatten()(l31)
#     l32 = Flatten()(l32)
#
#     conc_l = Concatenate()([l31, l32])
#
#     l4 = Dense(units=128, activation='relu')(conc_l)
#     l5 = Dense(units=64, activation='relu')(l4)
#
#     out1 = Dense(units=1, activation='sigmoid', name='arousal')(l5)
#     out2 = Dense(units=1, activation='sigmoid', name='valence')(l5)
#
#     return out1, out2

def stacked_LSTM(arch):
    if arch == 'FED':
        unit = 1
        activ = 'sigmoid'
    elif arch == 'CENT':
        unit = 2
        activ = 'softmax'

    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)
    x = LSTM(100, return_sequences=True)(x)
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(1000, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(units=64, activation='relu')(x)

    arousal = Dense(units=unit, activation=activ, name='arousal')(x)
    valence = Dense(units=unit, activation=activ, name='valence')(x)
    model = Model(inputs=inp, outputs={'arousal': arousal, 'valence': valence})

    return model


def autoencoder_LSTM(arch):
    if arch == 'FED':
        unit = 1
        activ = 'sigmoid'
    elif arch == 'CENT':
        unit = 2
        activ = 'softmax'

    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)

    x = LSTM(100, return_sequences=True)(x)
    x = LSTM(10, return_sequences=False)(x)
    x = RepeatVector(10)(x)
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = TimeDistributed(Dense(units=64, activation='relu'))(x)
    x = Flatten()(x)

    arousal = Dense(units=unit, activation=activ, name='arousal')(x)
    valence = Dense(units=unit, activation=activ, name='valence')(x)
    model = Model(inputs=inp, outputs={'arousal': arousal, 'valence': valence})

    return model


def bi_LSTM(arch):

    if arch == 'FED':
        unit = 1
        activ = 'sigmoid'
    elif arch == 'CENT':
        unit = 2
        activ = 'softmax'

    inp = Input(shape=(13055,))
    x = Reshape((13055, 1), input_shape=(13055,))(inp)
    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)

    arousal = Dense(units=unit, activation=activ, name='arousal')(x)
    valence = Dense(units=unit, activation=activ, name='valence')(x)
    model = Model(inputs=inp, outputs={'arousal': arousal, 'valence': valence})

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
