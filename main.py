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
from sklearn.utils import class_weight




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

    t_shape = 3038

    ###############################################################################################################
    ###############################################################################################################
    #                       CENT
    ###############################################################################################################
    ###############################################################################################################

    if self.arch == 'CENT':

      print('\nThe number of all stacked samples are', self.x.shape[0])
      if self.ml == 'CNN':
          self.model_cent = DNN.CNN(t_shape)
      elif self.ml == 'AUTOENCODER_LSTM':
          self.model_cent = DNN.AUTOENCODER_LSTM(t_shape)
      elif self.ml == 'STACKED_LSTM':
          self.model_cent = DNN.STACKED_LSTM(t_shape)
      elif self.ml == 'BI_LSTM':
          self.model_cent = DNN.BI_LSTM(t_shape)

      self.model_cent_aro = self.model_cent
      self.model_cent_val = self.model_cent

      self.model_cent_aro.compile(optimizer=self.optimizer,
                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                              metrics=[tf.keras.metrics.BinaryAccuracy()]
                              )

      self.model_cent_val.compile(optimizer=self.optimizer,
                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                              metrics=[tf.keras.metrics.BinaryAccuracy()]
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
          fed_model = DNN.CNN(t_shape)
      elif self.ml == 'AUTOENCODER_LSTM':
          fed_model = DNN.AUTOENCODER_LSTM(t_shape)
      elif self.ml == 'STACKED_LSTM':
          fed_model = DNN.STACKED_LSTM(t_shape)
      elif self.ml == 'BI_LSTM':
          fed_model = DNN.BI_LSTM(t_shape)


      self.g_model = fed_model
      self.g_model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                           metrics=[tf.keras.metrics.BinaryAccuracy()]
                           )

      # if self.arch == 'CNN':
      #   fed_model = EmoRec.reset_weights(fed_model)

      self.l_model = fed_model
      self.l_model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                           metrics=[tf.keras.metrics.BinaryAccuracy()]
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
    x = np.concatenate([xx, cwt.mean(axis=1), sf, ss,
                        resp.reshape(resp.shape[0], -1)], axis=1)

    return x, y


  def stacked_up_list_for_test(self, list):

    stacked_array = [self.stack_up(i) for i in list]
    x = [stacked_array[i][0] for i in range(len(stacked_array))]
    y = [stacked_array[i][1] for i in range(len(stacked_array))]

    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)
  # @staticmethod
  # def reset_weights(model):
  #   import tensorflow.keras.backend as K
  #
  #   for layer in model.layers:
  #     if hasattr(layer, 'kernel_initializer'):
  #       layer.kernel.initializer.run()
  #     if hasattr(layer, 'bias_initializer'):
  #       layer.bias.initializer.run()
  #
  #   return model


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

    history_aro = LossHistory()
    history_val = LossHistory()

    ###############################################################################################################
    ###############################################################################################################
    #                       CENT
    ###############################################################################################################
    ###############################################################################################################

    if self.arch == 'CENT':
      print(self.cwt.mean(axis=2).shape, self.cwt.mean(axis=1).shape, self.cwt.mean(axis=0).shape)
      x_dataset = np.concatenate([self.x, self.cwt.mean(axis=1), self.sf, self.ss,
                                  self.resp.reshape(self.resp.shape[0], -1)], axis=1)

      n_samples = x_dataset.shape[0]
      print('shapes', n_samples)

      # Training Testing samples Ratio
      # tr_te_rate_array = np.arange(0, 1, 0.1)

      test_size = round(0.1 * x_dataset.shape[0])
      print('test size ', test_size)



      def cross_validation(x_train, y_train, x_test, y_test):

          self.model_cent_aro.set_weights(self.model_cent.get_weights())
          self.model_cent_val.set_weights(self.model_cent.get_weights())

          class_weights = class_weight.compute_class_weight('balanced',
                                                            np.unique(y_train[:, 0].flatten()),
                                                            y_train[:, 0].flatten())

          print('class weights aro: ', class_weights)

          class_weights = class_weight.compute_class_weight('balanced',
                                                            np.unique(y_train[:, 1].flatten()),
                                                            y_train[:, 1].flatten())

          print('class weights VAL: ', class_weights)

          print('y train {}, test {}, and x train {}, test {} '.format(y_train.shape,
                                                                       y_test.shape,
                                                                       x_train.shape,
                                                                       x_test.shape))

          cls_weight ={0: 2.2,
                       1: 0.95}
          # Training
          self.model_cent_aro.fit(x=x_train, y=y_train[:, 0],
                                  batch_size=B, epochs=EPOCH, verbose=1,
                                  class_weight=cls_weight, callbacks=[history_aro])

          cls_weight ={0: 2.7,
                       1: 0.85}

          self.model_cent_val.fit(x=x_train, y=y_train[:, 1],
                                  batch_size=B, epochs=EPOCH, verbose=1,
                                  class_weight=cls_weight, callbacks=[history_val])



          # Testing
          y_hat_aro = self.model_cent_aro.predict(x=x_test,  batch_size=B)
          y_hat_val = self.model_cent_val.predict(x=x_test,  batch_size=B)
          y_hat = [y_hat_aro, y_hat_val]


          print(np.array(y_hat).shape)
          ut.report(y_test, y_hat, self.arch, self.ml,
                    [self.model_cent_aro, self.model_cent_val],
                    [history_aro, history_val])

      k_fold = 10
      x_test, x_train = np.split(x_dataset.copy(), [test_size], axis=0)
      y_test, y_train = np.split(self.y.copy(), [test_size], axis=0)
      print('indices of used test fold {}-{}'.format(0, test_size))
      print('train and test shape', x_train.shape, x_test.shape)

      for i in range(1, k_fold-1):

        cross_validation(x_train, y_train, x_test, y_test)

        print('indices of used test fold {}-{}'.format((i-1)*test_size, i*test_size))
        print('train and test shape', x_train.shape, x_test.shape)
        x_train[(i-1)*test_size:i*test_size], x_test = x_test, x_train[i*test_size:(i+1)*test_size].copy()
        y_train[(i-1)*test_size:i*test_size], y_test = y_test, y_train[i*test_size:(i+1)*test_size].copy()




    ###############################################################################################################
    ###############################################################################################################
    #                       FED
    ###############################################################################################################
    ###############################################################################################################

    elif self.arch == 'FED':


      def separ(name, cls, indx, test_data, tarin_list):

          x_test, y_test = test_data

          for ge in range(EPOCH):

            print('\nGlobal Epoch {} .'.format(ge))

            self.l_model.set_weights(self.g_model.get_weights())

            selected_for_training = np.random.choice(tarin_list, size=self.P,
                                                     replace=False).tolist()  # we can set p here to consider a weight for each mdoel
            print(selected_for_training)

            temp_l_models_weight_list = []
            x_tribute, y_tribute = [], []

            for ith in selected_for_training:
                print(' Training {}-th user on its local dataset.'.format(ith))
                self.l_model.set_weights(self.g_model.get_weights())

                x, y = self.stack_up(ith)  # the i-the user's data is extracting

                x_tribute.append(x[:300])
                y_tribute.append(y[:300])

                # y_arousal = to_categorical(te_data[:, 0], 2)
                # y_valence = to_categorical(te_data[:, 1], 2)


                self.l_model.fit(x=x, y=y[:, indx],
                                 batch_size=B, epochs=LE, verbose=0)

                temp_l_models_weight_list.append(self.l_model.get_weights())

            y_tribute = np.concatenate(y_tribute, axis=0)
            x_tribute = np.concatenate(x_tribute, axis=0)
            print('tribute shape ', x_tribute.shape, y_tribute.shape)

            rand_models_weights_for_global_avg = np.random.choice(len(temp_l_models_weight_list),
                                                                  size=self.C,
                                                                  replace=False).tolist()  # we can set p here to consider a weight for each mdoel

            rand_models_weights_for_global_avg = [temp_l_models_weight_list[i] for i in rand_models_weights_for_global_avg]

            self.g_model.set_weights(self.get_average_weights(rand_models_weights_for_global_avg))


            self.g_model.fit(x=x_tribute, y=y_tribute[:, indx], epochs=LE, verbose=1, batch_size=B, class_weight=cls)


          y_hat = self.g_model.predict(x=x_test,  # , self.cwt_te, self.sf_te, self.ss_te, self.resp_te],
                                       batch_size=B)

          ut.fed_report(y_test[:, indx], y_hat, self.arch, self.ml, name)


      users_list = np.arange(0, 30)
      k_fold = 11
      test_size = 3
      fed_test_list, fed_train_list = np.split(users_list, [test_size], axis=0)

      for i in range(1, k_fold):

          # cls_weight = {0: 2.2,
          #               1: 0.95}
          print('Training Set (user ide)', fed_train_list)
          print('Test Set (user ide)', fed_test_list)

          cls_weight = {0: 2.,
                        1: 1.}

          separ('arousal', cls_weight, 0,
                self.stacked_up_list_for_test(fed_test_list),
                fed_train_list)

          cls_weight = {0: 2.,
                        1: 1.}

          separ('valence', cls_weight, 1,
                self.stacked_up_list_for_test(fed_test_list),
                fed_train_list)

          if i < 10:
            fed_test_list, fed_train_list[(i-1) * test_size: i * test_size] = fed_train_list[(i-1) * test_size:(i) * test_size].copy(), fed_test_list



if __name__ == '__main__':
  print('Starting ... \n')


  attr = {'gsr_only': True,
          'decompose': True,
          'minmax_norm': True,
          'architecture': 'FED',
          'model': 'CNN',
          'C': 15,
          'P': 25,
          }


  obj = EmoRec(attr)
  obj.train_and_test(EPOCH=5, LE=3)


  #
  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'AUTOENCODER_LSTM',
  #         'C': 4,
  #         'P': 5,
  #         }
  #
  #
  # obj = EmoRec(attr)
  # obj.train_and_test(EPOCH=5, LE=1)
  #
  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'BI_LSTM',
  #         'C': 4,
  #         'P': 5,
  #         }
  #
  #
  # obj = EmoRec(attr)
  # obj.train_and_test(EPOCH=5, LE=1)
  #
  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'STACKED_LSTM',
  #         'C': 4,
  #         'P': 5,
  #         }
  #
  #
  # obj = EmoRec(attr)
  # obj.train_and_test(EPOCH=5, LE=1)