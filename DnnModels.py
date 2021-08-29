import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, Dense, Concatenate
from tensorflow.keras.layers import LSTM, Bidirectional, MaxPooling2D
# from tensorflow.keras.layers import ConvLSTM1D, ConvLSTM2D, BatchNormalization


class DNN:
    def __init__(self, ml, input):

      self._input = input

    def CNN(self):

        l11 = Conv1D(32, kernel_size=3, padding='same', strides=1, activation='relu')(self._input[0])
        l21 = Conv1D(32, kernel_size=3, padding='same', strides=1, activation='relu')(l11)
        # print('first input ', l1.shape, l2.shape)

        l12 = Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu')(self._input[1])
        l22 = Conv2D(32, kernel_size=3, padding='same', strides=1, activation='relu')(l12)

        l1 = Flatten()(l21)
        l3 = Flatten()(l22)

        conc_l = Concatenate()([l1, l3])

        l7 = Dense(units=128, activation='relu')(conc_l)
        l8 = Dense(units=64, activation='relu')(l7)

        out1 = Dense(units=1, activation='sigmoid', name='arousal')(l8)
        out2 = Dense(units=1, activation='sigmoid', name='valence')(l8)

        return out1, out2

    # def conv_LSTM(self):
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

    def stacked_LSTM(self):

        l11 = LSTM(1000, return_sequences=True)(self._input[0])
        l12 = LSTM(1000, return_sequences=True)(self._input[1])

        l11 = Dropout(0.2)(l11)
        l12 = Dropout(0.2)(l12)

        l21 = LSTM(1000)(l11)
        l22 = LSTM(1000)(l12)

        out1 = Dense(units=1, activation='sigmoid', name='arousal')(l21)
        out2 = Dense(units=1, activation='sigmoid', name='valence')(l22)

        return out1, out2

    def LSTM(self):

        l11 = LSTM(1000, return_sequences=True)(self._input[0])
        l12 = LSTM(1000, return_sequences=True)(self._input[1])

        l11 = Dropout(0.2)(l11)
        l12 = Dropout(0.2)(l12)

        l11 = Flatten()(l11)
        l12 = Flatten()(l12)

        conc_l = Concatenate()([l11, l12])

        l3 = Dense(units=128, activation='relu')(conc_l)
        l4 = Dense(units=64, activation='relu')(l3)

        out1 = Dense(units=1, activation='sigmoid', name='arousal')(l4)
        out2 = Dense(units=1, activation='sigmoid', name='valence')(l4)

        return out1, out2

    def bi_LSTM(self):

        l11 = Bidirectional(LSTM(1000, return_sequences=True))(self._input[0])
        l12 = Bidirectional(LSTM(1000, return_sequences=True))(self._input[1])

        l11 = Dropout(0.2)(l11)
        l12 = Dropout(0.2)(l12)

        l11 = Flatten()(l11)
        l12 = Flatten()(l12)

        conc_l = Concatenate()([l11, l12])

        l3 = Dense(units=128, activation='relu')(conc_l)
        l4 = Dense(units=64, activation='relu')(l3)

        out1 = Dense(units=1, activation='sigmoid', name='arousal')(l4)
        out2 = Dense(units=1, activation='sigmoid', name='valence')(l4)

        return out1, out2

    def unsequenced_LSTM(self):

        l11 = LSTM(1000)(self._input[0])
        l12 = LSTM(1000)(self._input[1])

        conc_l = Concatenate()([l11, l12])
        l3 = Dense(units=8, activation='relu')(conc_l)

        out1 = Dense(units=1, activation='sigmoid', name='arousal')(l3)
        out2 = Dense(units=1, activation='sigmoid', name='valence')(l3)

        return out1, out2


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
