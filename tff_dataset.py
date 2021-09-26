import numpy as np
import collections
import dataset as dt
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import reshape, nest, config
from tensorflow.keras.utils import to_categorical


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dropout, Input
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Reshape
from tensorflow.keras.layers import LSTM, Bidirectional, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization  # , ConvLSTM1D

path = './GSR/CASE_dataset/interpolated/{}/'
phy_dir = path.format('physiological')
ann_dir = path.format('annotations')



def tff_dataset(usr_data_set):

    split = len(usr_data_set)
    client_train_dataset = collections.OrderedDict()

    for i in range(0, split):
        client_name = "client_" + str(i)
        x, y = usr_data_set[i]

        arousal = y[:, 0]
        valence = y[:, 1]

        data = collections.OrderedDict((('x', x),
                                        ('y', collections.OrderedDict((('arousal', arousal),
                                                                       ('valence', valence))))))


        client_train_dataset[client_name] = data

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

    sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
    sample_element = next(iter(sample_dataset))

    def preprocess(dataset):
        NUM_EPOCHS = 5
        BATCH_SIZE = 32
        PREFETCH_BUFFER = 10

        def batch_format_fn(element):

            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
            return collections.OrderedDict(
                x=reshape(element['x'], [-1, 13055]),
                y=collections.OrderedDict((('arousal', reshape(element['y']['arousal'], [-1, 1])),
                                           ('valence', reshape(element['y']['valence'], [-1, 1])))))


        return dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE).map(
            batch_format_fn).prefetch(PREFETCH_BUFFER)

    preprocessed_sample_dataset = preprocess(sample_dataset)

    def make_federated_data(client_data, client_ids):
        return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

    federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)


    return federated_train_data, preprocessed_sample_dataset












