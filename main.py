import os
import numpy as np
import pandas as pd
import dataset as dt
import tensorflow as tf

import utils as ut
import DnnModels as DNN
from DnnModels import LossHistory
# from sklearn.utils import class_weight
from functools import partial

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


np.random.seed(10)  # for reproducibility
pd.options.display.width = 0

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class EmoRec:

  def __init__(self, kwargs):

    print('The DNN class is building ....')
    # cur_path = os.getcwd()
    if 'posix' in os.name:
      path = './GSR/CASE_dataset/interpolated/{}/'
    elif 'nt' in os.name:
      path = '/home/amiir/Desktop/FedGSR/GSR/CASE_dataset/interpolated/{}/'

    # initializing values
    self.phy_dir = kwargs.get('phy_dir', path.format('physiological'))
    self.ann_dir = kwargs.get('ann_dir', path.format('annotations'))
    self.arch = kwargs.get('architecture', 'CENT')
    self.ml = kwargs.get('model', 'CNN')
    self.decompose_flag = kwargs.get('decompose', False)
    self.gsr_only_flag = kwargs.get('gsr_only', False)
    self.minmax = kwargs.get('minmax_norm', False)
    self.ecg_flag = kwargs.get('ecg', False)
    self.lr = kwargs.get('lr', 0.001)
    self.optimizer = kwargs.get('optimizer', Adam(self.lr))
    "TODO: WE CAN CONSIDER OTHER FEATURES SAME AS THE ABOVE FEATURES WITH SAME METHOD"

    # Dataset class is called and built here.
    # The load_data() function gives us Phasic GSR (x),
    # features extracted by Continuous wavelet transform (cwt) from Phasic GSR,
    # and labels (y)
    dataset = dt.CASE(self.phy_dir, self.ann_dir, self.arch)
    self.x, self.y, self.cwt, self.sf, self.ss, self.resp = dataset.load_data() # pass address here to the files.
    print('The Dataset is loaded.')
    print('shapes are ')
    print(self.x.shape, self.y.shape, self.cwt.shape, self.sf.shape, self.ss.shape, self.resp.shape)

    # self.Num_Usr is the number of file in the dataset
    self.Num_Usr = self.x.shape[0]
    self.C = kwargs.get('C', self.x.shape[0]) # Number of selected model for Aggregating (Must be <29)
    self.P = kwargs.get('P', self.x.shape[0]) # Number of selected model for training (Must be <29)
    if self.arch == 'FED':
      self.Num_Sess = kwargs.get('Num_Sess', self.x.shape[1])

    print('In this run, we use a {}-based model with {} architecture.'.format(self.ml, self.arch))





    w1 = np.ones((2, 2))
    w1[1, 0] = 2.6
    w1[1, 1] = 2.6

    w2 = np.ones((2, 2))
    w2[1, 0] = 1.3
    w2[1, 1] = 1.3

    loss1 = ut.weighted_categorical_crossentropy(w1)
    loss2 = ut.weighted_categorical_crossentropy(w2)

    loss1.__name__ = 'loss1'
    loss2.__name__ = 'loss2'

    losses = {'arousal': loss1,
              'valence': loss2,
              }

    ###############################################################################################################
    ###############################################################################################################
    #                       CENT
    ###############################################################################################################
    ###############################################################################################################



    if self.arch == 'CENT':
      print('\nThe number of all stacked samples are', self.x.shape[0])

      if self.ml == 'CNN':
          self.model_cent = DNN.CNN()
      elif self.ml == 'autoencoder_LSTM':
          self.model_cent = DNN.AUTOENCODER_LSTM()
      elif self.ml == 'stacked_LSTM':
          self.model_cent = DNN.STACKED_LSTM()
      elif self.ml == 'bi_LSTM':
          self.model_cent = DNN.BI_LSTM()


      self.model_cent.compile(optimizer=self.optimizer,
                              loss=losses,
                              metrics=[tf.keras.metrics.CategoricalAccuracy()]
                              )



    ###############################################################################################################
    ###############################################################################################################
    #                       FED
    ###############################################################################################################
    ###############################################################################################################
    elif self.arch == 'FED':
      print('\nThe Number of Users is {}, The Number of sessions for each user is {}.'.format(self.Num_Usr,
                                                                                              self.Num_Sess))
      print('The Number of Users used for training over local data (P) is {} '.format(self.P))
      print('The Number of Users used for aggregation in the global model (C) is {} '.format(self.C))
      self.fed_history = []

      if self.ml == 'CNN':
        fed_model = DNN.CNN(self.arch)
      elif self.ml == 'autoencoder_LSTM':
        fed_model = DNN.autoencoder_LSTM(self.arch)
      elif self.ml == 'stacked_LSTM':
        fed_model = DNN.stacked_LSTM(self.arch)
      elif self.ml == 'bi_LSTM':
        fed_model = DNN.bi_LSTM(self.arch)


      self.g_model = fed_model
      self.g_model.compile(optimizer=self.optimizer,
                           loss=losses,
                           metrics=[tf.keras.metrics.categorical_accuracy()]
                            )

      # if self.arch == 'CNN':
      #   fed_model = EmoRec.reset_weights(fed_model)

      self.l_model = fed_model
      self.l_model.compile(optimizer=self.optimizer,
                           loss=losses,
                           metrics=[tf.keras.metrics.categorical_accuracy()]
                           )




  @staticmethod
  def get_average_weights(weights):

    # weights = [model.get_weights() for model in models]
    new_weights = list()

    for weights_list_tuple in zip(*weights):
      new_weights.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))

    return new_weights

  def stack_up(self, ith_usr):
    """

    This function stacks up all the sessions array in the I-th user in one sequential array.
    It is just used in FED architecture
    If we do this function, it means our framework is using a IID format of a dataset.

    :param ith_usr: The number of user's place in the row of 30 users in the dataset
    :return: return sequential expanded-dimension array of x, y, cwt

    """
    xx = np.concatenate([self.x[ith_usr][i] for i in range(self.Num_Sess)])
    y = np.concatenate([self.y[ith_usr][i] for i in range(self.Num_Sess)])
    cwt = np.concatenate([self.cwt[ith_usr][i] for i in range(self.Num_Sess)])
    sf = np.concatenate([self.sf[ith_usr][i] for i in range(self.Num_Sess)])
    ss = np.concatenate([self.ss[ith_usr][i] for i in range(self.Num_Sess)])
    resp = np.concatenate([self.resp[ith_usr][i] for i in range(self.Num_Sess)])

    # if 'LSTM' not in self.ml:
    # np.expand_dims(cwt, axis=-1)
    x = np.concatenate([xx, cwt.reshape(cwt.shape[0], -1), sf, ss, resp.reshape(resp.shape[0], -1)], axis=1)

    return x, y

  @staticmethod
  def reset_weights(model):
    import tensorflow.keras.backend as K

    for layer in model.layers:
      if hasattr(layer, 'kernel_initializer'):
        layer.kernel.initializer.run()
      if hasattr(layer, 'bias_initializer'):
        layer.bias.initializer.run()

    return model


  def train_and_test(self, B=32, EPOCH=5, LE=2):
    """
    This function Trains the model.

    :param B: B represents the number of sample in a batch for each iteration
    :param EPOCH: EPOCH is showing either Global Epochs in FED architecture or
               the number of Epochs in CENT architecture.
    :param LE: The number of Local Epochs each client makes over its local dataset.
               It is used just for the FED architecture.
    :return: None

    """
    print('\nTraining Phase is starting. It may take a while to train the model.')


    if self.arch == 'CENT':

      x_dataset = np.concatenate([self.x, self.cwt.reshape(self.cwt.shape[0], -1),
                                  self.sf, self.ss, self.resp.reshape(self.resp.shape[0], -1)], axis=1)
      print(x_dataset.shape)

      # Training Testing samples Ratio
      tr_te_rate = round(0.8 * x_dataset.shape[0])
      print('Number of Samples: ', x_dataset.shape[0], ' and Number of training Samples: ', tr_te_rate)
      print('Number of testing Samples: ', x_dataset.shape[0] - tr_te_rate)
      # Training and Testing samples
      x_train = x_dataset[: tr_te_rate]
      y_train = self.y[: tr_te_rate]
      x_test = x_dataset[tr_te_rate:]
      y_test = self.y[tr_te_rate:]




      print('y train {},test {},   and x train {}, test {} '.format(y_train.shape, y_test.shape, x_train.shape, x_test.shape))

      # Training
      self.model_cent.fit(x=x_train, y=[to_categorical(y_train[:, 0], 2), to_categorical(y_train[:, 1], 2)],
                          batch_size=B, epochs=EPOCH, verbose=1)

      # Testing
      y_hat = self.model_cent.predict(x=x_test,  batch_size=B)
      print(np.array(y_hat).shape)
      ut.report(y_test, y_hat, self.arch, self.ml)


    elif self.arch == 'FED':

      test_user = self.Num_Usr - 1  # the last user in the array of users will be used for testing phase.
      x_test, y_test = self.stack_up(test_user)

      weights_ = self.l_model.get_weights()

      for ge in range(EPOCH):

        print('\nGlobal Epoch {} .'.format(ge))

        self.l_model.set_weights(self.g_model.get_weights())

        selected_for_training = np.random.choice(range(self.Num_Usr - 1), size=self.P,
                                                 replace=False).tolist()  # we can set p here to consider a weight for each mdoel
        print(selected_for_training)

        temp_save_weight = []
        tr_data, te_data = [], []
        for ith in selected_for_training:
          print(' {}-th user for training on local dataset.'.format(ith))
          self.l_model.set_weights(self.g_model.get_weights())

          x, y = self.stack_up(ith)  # the i-the user's data is extracting
          tr_data.append(x[:20]), te_data.append(y)

          y_arousal = to_categorical(te_data[:, 0], 2)
          y_valence = to_categorical(te_data[:, 1], 2)

          self.l_model.fit(x=x, y=[y_arousal, y_valence],
                           batch_size=B, epochs=LE, verbose=2)

          temp_save_weight.append(self.l_model.get_weights())


        rand_models_weights_for_global_avg = np.random.choice(len(temp_save_weight), size=self.C,
                                                              replace=False).tolist()  # we can set p here to consider a weight for each mdoel

        rand_models_weights_for_global_avg = [temp_save_weight[i] for i in rand_models_weights_for_global_avg]

        self.g_model.set_weights(self.get_average_weights(rand_models_weights_for_global_avg))


        self.g_model.fit(x=x[:100], y={"arousal": y[:100, 0], "valence": y[:100, 1]},
                         verbose=1, batch_size=B)


        y_hat = self.g_model.predict(x=x_test,  # , self.cwt_te, self.sf_te, self.ss_te, self.resp_te],
                                     batch_size=B)
        ut.report(y_test, y_hat, self.arch, self.ml)


if __name__ == '__main__':
  print('Starting ... \n')


  attr = {'gsr_only': True,
          'decompose': True,
          'minmax_norm': True,
          'architecture': 'CENT',
          'model': 'CNN',
          'C': 4,
          'P': 5,
          }


  obj = EmoRec(attr)
  obj.train_and_test(EPOCH=1, LE=1)



  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'stacked_LSTM',
  #         }
  #
  #
  #
  # obj = EmoRec(attr)
  # obj.train_and_test(EPOCH=1)
  #
  #
  #
  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'autoencoder_LSTM',
  #         }
  #
  #
  #
  # obj = EmoRec(attr)
  # obj.train_and_test(EPOCH=1)
  #
  #
  #
  #
  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'bi_LSTM',
  #         }
  #
  #
  #
  # obj = EmoRec(attr)
  # obj.train_and_test(EPOCH=1)