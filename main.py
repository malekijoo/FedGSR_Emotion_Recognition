import os
import numpy as np
import pandas as pd
import dataset as dt
import tensorflow as tf

import utils as ut
from DnnModels import DNN as DNN
from DnnModels import LossHistory
# from sklearn.utils import class_weight
from functools import partial

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



np.random.seed(10)  # for reproducibility
pd.options.display.width = 0



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
    print(self.phy_dir)
    self.arch = kwargs.get('architecture', 'CENT')
    self.ml = kwargs.get('model', 'CNN')
    self.decompose_flag = kwargs.get('decompose', False)
    self.gsr_only_flag = kwargs.get('gsr_only', False)
    self.minmax = kwargs.get('minmax_norm', False)
    self.ecg_flag = kwargs.get('ecg', False)
    self.lr = kwargs.get('lr', 0.0001)
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


    if self.arch == 'CENT':
      print('\nThe number of all stacked samples are', self.x.shape[0])
      self.model = self._create_model()

    elif self.arch == 'FED':
      print('\nThe Number of Users is {}, The Number of sessions for each user is {}.'.format(self.Num_Usr,
                                                                                              self.Num_Sess))
      print('The Number of Users used for training over local data (P) is {} '.format(self.P))
      print('The Number of Users used for aggregation in the global model (C) is {} '.format(self.C))
      self.fed_history = []


      # self.model = [self._create_model() for num_usr in range(3)] # this line will be deleted
      # self.model = [self._create_model() for num_usr in range((self.Num_Usr-1))]
      self.model = self._create_model()
      # print('Number of models used for aggregating the global model is  {}.'.format(len(self.model)))
      self.global_model = self._create_model()



  def _create_model(self):


    if self.gsr_only_flag:
      input_0 = Input(shape=(1000, 1), name="input_0")
      input_1 = Input(shape=(1000, 8, 1), name="input_1")  # continuous wavelet transform, cwt
      input_2 = Input(shape=(51, 1), name="input_2")  # spectral flux, sf
      input_3 = Input(shape=(4, 1), name="input_3")  # statistic features, ss
      input_4 = Input(shape=(1000, 4), name="input_4")  # responses, SCR_Peaks, SCR_RiseTime, SCR_Height, SCR_Recovery

      if 'LSTM' in self.ml:
        input_1 = Input(shape=(1000, 7), name="input_1")


    else:
      'TODO: it should be determinded how many features we are going to work on.'
      pass



    dnn = DNN(self.ml, [input_0, input_1, input_2,input_3, input_4])


    if self.ml == 'CNN':
      arousal, valence = dnn.CNN()
    elif self.ml == 'LSTM':
      arousal, valence = dnn.LSTM()
    elif self.ml == 'conv_LSTM':
      arousal, valence = dnn.conv_LSTM()
    elif self.ml == 'stacked_LSTM':
      arousal, valence = dnn.stacked_LSTM()
    elif self.ml == 'bi_LSTM':
      arousal, valence = dnn.bi_LSTM()
    elif self.ml == 'unsequenced_LSTM':
      arousal, valence = dnn.unsequenced_LSTM()

    w1 = np.ones((2, 2))
    w1[1, 0] = 2
    w1[1, 1] = 2

    w2 = np.ones((2, 2))
    w2[1, 0] = 2
    w2[1, 1] = 2

    loss1 = partial(ut.weighted_categorical_crossentropy, weights=w1)
    loss2 = partial(ut.weighted_categorical_crossentropy, weights=w2)

    loss1.__name__ = 'loss1'
    loss2.__name__ = 'loss2'

    losses = {'arousal': loss1,
              'valence': loss2,}


    # losses = {'arousal': 'binary_crossentropy',
    #           'valence': 'binary_crossentropy',}

    metrics = {'arousal': 'accuracy',
               'valence': 'accuracy',}

    model = Model(inputs=[input_0, input_1, input_2, input_3, input_4],
                  outputs=[arousal, valence],)
    # plot_model(self.cnn, to_file='CNN.png', show_shapes=True, show_layer_names=True)
    model.compile(optimizer=self.optimizer,
                  loss=losses,
                  metrics=metrics)


    return model




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
    x = np.concatenate([self.x[ith_usr][i] for i in range(self.Num_Sess)])
    y = np.concatenate([self.y[ith_usr][i] for i in range(self.Num_Sess)])
    cwt = np.concatenate([self.cwt[ith_usr][i] for i in range(self.Num_Sess)])
    sf = np.concatenate([self.sf[ith_usr][i] for i in range(self.Num_Sess)])
    ss = np.concatenate([self.ss[ith_usr][i] for i in range(self.Num_Sess)])
    resp = np.concatenate([self.resp[ith_usr][i] for i in range(self.Num_Sess)])

    if 'LSTM' not in self.ml:
      cwt = np.expand_dims(cwt, axis=-1)

    return np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), cwt,\
           np.expand_dims(sf, axis=-1), np.expand_dims(ss, axis=-1), np.expand_dims(resp, axis=-1)


  def train(self, B=32, GE=10, LE=1):
    """
    This function Trains the model.

    :param B: B represents the number of sample in a batch for each iteration
    :param GE: GE is showing either Global Epochs in FED architecture or
               the number of Epochs in CENT architecture.
    :param LE: The number of Local Epochs each client makes over its local dataset.
               It is used just for the FED architecture.
    :return: None

    """
    print('\nTraining Phase is starting. It may take a while to train the model.')

    history = LossHistory()

    # class_weights = {'arousal': {0: 2, 1: 0.2}, 'valence': {0: 2, 1: 0.2}}
    #
    # print('class weights: ', class_weights)

    if self.arch == 'CENT':

      self.x = np.expand_dims(self.x, axis=-1)
      # self.y = np.expand_dims(self.y, axis=-1)
      self.sf = np.expand_dims(self.sf, axis=-1)
      self.ss = np.expand_dims(self.ss, axis=-1)
      self.resp = np.expand_dims(self.resp, axis=-1)

      if 'LSTM' not in self.ml:
        self.cwt = np.expand_dims(self.cwt, axis=-1)

      assert self.x.shape[0] == self.y.shape[0] == self.cwt.shape[0]
      assert self.sf.shape[0] == self.ss.shape[0] == self.resp.shape[0]

      # Training Testing samples Ratio
      tr_te_rate = round(0.9 * self.x.shape[0])

      print('Number of Samples: ', self.x.shape[0], ' and Number of training Samples: ', tr_te_rate)
      print('Number of testing Samples: ', self.x.shape[0] - tr_te_rate)

      # Training samples
      self.x_tr, self.y_tr, self.cwt_tr = self.x[:tr_te_rate], self.y[:tr_te_rate], self.cwt[:tr_te_rate]
      self.sf_tr, self.ss_tr, self.resp_tr = self.sf[:tr_te_rate], self.ss[:tr_te_rate], self.resp[:tr_te_rate]

      # Testing samples
      self.x_te, self.y_te, self.cwt_te = self.x[tr_te_rate:], self.y[tr_te_rate:], self.cwt[tr_te_rate:]
      self.sf_te, self.ss_te, self.resp_te = self.sf[tr_te_rate:], self.ss[tr_te_rate:], self.resp[tr_te_rate:]

      # y_arousal = to_categorical(self.y_tr[:, 0], 2)
      # y_valence = to_categorical(self.y_tr[:, 0], 2)
      #
      self.model.fit(x=[self.x_tr, self.cwt_tr, self.sf_tr, self.ss_tr, self.resp_tr],
                     y={"arousal": to_categorical(self.y_tr[:, 0], 2), "valence": to_categorical(self.y_tr[:, 1], 2)},
                     batch_size=B, epochs=GE, verbose=1, callbacks=[history])



    elif self.arch == 'FED':

      test_user = self.Num_Usr-1 # the last user in the array of users will be used for testing phase.
      self.x_te, self.y_te, self.cwt_te, self.sf_te, self.ss_te, self.resp_te = self.stack_up(test_user)

      weights_ = self.model.get_weights()
      temp_save_weight = []


      for ge in range(GE):
        print('\nGlobal Epoch {} .'.format(ge))

        self.model.set_weights(self.global_model.get_weights())

        selected_for_training = np.random.choice(range(self.Num_Usr-1), size=self.P, replace=False).tolist() # we can set p here to consider a weight for each mdoel

        for ith in selected_for_training:
          self.model.set_weights(self.global_model.get_weights())

          x, y, cwt, sf, ss, resp = self.stack_up(ith) # the i-the user's data is extracting
          # print('shapes after def stack_up() ', x.shape, y.shape, cwt.shape)
          self.model.fit(x=[cwt, sf, ss, resp], y={"arousal": y[:, 0], "valence": y[:, 1]},
                         batch_size=B, epochs=LE, verbose=2)

          temp_save_weight.append(self.model.get_weights())

          self.model.set_weights(self.global_model.get_weights())

        rand_models_weights_for_global_avg = np.random.choice(len(temp_save_weight),
                                                              size=self.C,
                                                              replace=False).tolist() # we can set p here to consider a weight for each mdoel

        rand_models_weights_for_global_avg = [temp_save_weight[i] for i in rand_models_weights_for_global_avg]

        self.global_model.set_weights(self.get_average_weights(rand_models_weights_for_global_avg))

        results = self.global_model.evaluate(x=[self.x_tr, self.cwt_te, self.sf_te, self.ss_te, self.resp_te],
                                             y={"arousal": self.y_te[:, 0],
                                                "valence": self.y_te[:, 1]},
                                             batch_size=B)
        self.fed_history.append(results)
        print('result is ', results)

  def test(self, B=32):
    """
    This function tests the model and print the confusion matrix
    and classification report based on the prediction of the
    trained model.

    :param B: Batch Size

    """
    print('\nTesting Phase is starting. The result and confusion matrix will be shown here')


    if self.arch == 'CENT':
      trained_model = self.model
      history = trained_model.history
      print('cent ', trained_model)

    elif self.arch == 'FED':
      trained_model = self.global_model
      history = self.fed_history


    y_hat = trained_model.predict(x=[self.x_te, self.cwt_te, self.sf_te, self.ss_te, self.resp_te], batch_size=B)

    ut.report(self.y_te, y_hat, self.arch, self.ml)
    ut.plots(history, self.arch, self.ml, name='main_model')



if __name__ == '__main__':
  print('Starting ... \n')

  # attr = {'phy_dir': physiological_dir,
  #         'ann_dir': annotation_dir,
  #         'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'CNN',
  #         'C': 2}


  attr = {'gsr_only': True,
          'decompose': True,
          'minmax_norm': True,
          'architecture': 'CENT',
          'model': 'CNN',
          }


  # attr = {'gsr_only': True,
  #         'decompose': True,
  #         'minmax_norm': True,
  #         'architecture': 'CENT',
  #         'model': 'CNN',
  #         'C': 5,
  #         'P': 8
  #         }


  obj = EmoRec(attr)
  obj.train(GE=10, LE=1)
  obj.test()







