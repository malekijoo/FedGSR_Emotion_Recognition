import os
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from itertools import product

from datetime import datetime
from numpy.lib import stride_tricks

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if 'posix' in os.name:
    deli = '/'
    cur_path = os.getcwd()
elif 'nt' in os.name:
    deli = '\\'
    cur_path = os.getcwd()



def report(y, y_hat, arch, ml):\

    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

    wd = cur_path + deli + 'report' + deli
    if not os.path.isdir(wd):
        os.makedirs(wd, exist_ok=True)
    save_path = wd + date_time + deli
    os.makedirs(save_path, exist_ok=True)

    def save_and_print_fn(y_, yhat, name):

        conf_mat = confusion_matrix(y_.tolist(), yhat.tolist())
        report_arousal = classification_report(y_.tolist(), yhat.tolist(), output_dict=True)
        df = pd.DataFrame(report_arousal)

        print('\nReport of {}'.format(name))
        print(classification_report(y_.tolist(), yhat.tolist()))
        print('\nConfusion Matrix')
        print(conf_mat)

        # Saving
        conf_mat_df = pd.DataFrame(conf_mat)
        conf_mat_df.to_csv(save_path + 'confusion_mat_{}_{}_{}.csv'.format(name, arch, ml), index=False)
        df.to_csv(save_path + 'rep_{}_{}_{}.csv'.format(name, arch, ml), index=False)
        print('Arousal and Valence saved in the report directory with postfix date and time ,', date_time)

    yhat_arousal = np.argmax(y_hat[0], axis=1)
    yhat_valence = np.argmax(y_hat[1], axis=1)
    y_arousal = np.squeeze(y[:, 0])
    y_valence = np.squeeze(y[:, 1])

    print(yhat_arousal.shape, yhat_valence.shape, y_arousal.shape, y_valence.shape)

    save_and_print_fn(y_arousal, yhat_arousal, 'arousal')
    save_and_print_fn(y_valence, yhat_valence, 'valence')



def plots(history, arch, ml, name):
    wd = cur_path + deli + 'figs' + deli

    if not os.path.isdir(wd):
      os.makedirs(wd, exist_ok=True)

    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

    save_path = wd + date_time + deli
    print('History plots are saved in : ', save_path)
    os.makedirs(save_path, exist_ok=True)

    if arch == 'CENT':

      plt.plot(history.history['loss'], 'm', label='Loss')
      plt.plot(history.history['arousal_loss'], 'r', label='Arousal Loss')
      plt.plot(history.history['valence_loss'], 'b', label='Valence Loss')
      plt.title('Classification Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend(numpoints=1, fontsize=20)
      plt.savefig(save_path+'Loss_{}_{}_{}.png'.format(name, arch, ml))
      plt.close()


      plt.plot(history.history['arousal_accuracy'], 'r', label='Arousal Accuracy')
      plt.plot(history.history['valence_accuracy'], 'b', label='Valence Accuracy')
      plt.title('Classification Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend(numpoints=1, fontsize=20)
      plt.savefig(save_path+'Accuracy_model({})_{}_{}.png'.format(name, arch, ml))

    elif arch == 'FED':

      plt.plot(history[:, 0], 'm', label='Loss')
      plt.plot(history[:, 1], 'r', label='Arousal Loss')
      plt.plot(history[:, 2], 'b', label='Valence Loss')
      plt.title('Classification Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend(numpoints=1, fontsize=20)
      # plt.show()
      plt.savefig(save_path + 'Loss_{}_{}_{}.png'.format(name, arch, ml))
      plt.close()

      plt.plot(history[:, 3], 'r', label='Arousal Accuracy')
      plt.plot(history[:, 4], 'b', label='Valence Accuracy')
      plt.title('Classification Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend(numpoints=1, fontsize=20)
      plt.savefig(save_path + 'Accuracy_model({})_{}_{}.png'.format(name, arch, ml))
      # lgnd = plt.legend(loc="lower left", numpoints=1, fontsize=20)
      # lgnd.legendHandles[0]._legmarker.set_markersize(20)



class weighted_categorical_crossentropy(tf.keras.losses.Loss):

    def __init__(self, weight, reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_categorical_crossentropy'):
        super().__init__(reduction=reduction, name=name)

        self.weight = weight

    def __call__(self, y_true, y_pred, **kwargs):

        # print(y_true.dtype, y_pred.dtype, self.weight.dtype)
        y_true = tf.cast(y_true, tf.float32)
        # print(y_true.dtype, y_pred.dtype, self.weight.dtype)

        nb_cl = len(self.weight)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())

        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (self.weight[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

