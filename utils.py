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



def report(y, y_hat, arch, ml, model, history):\

    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

    wd = cur_path + deli + 'report' + deli
    if not os.path.isdir(wd):
        os.makedirs(wd, exist_ok=True)
    save_path = wd + date_time + deli
    os.makedirs(save_path, exist_ok=True)

    def plots(history_aro, history_val):

        print('History plots are saved in : ', save_path)
        os.makedirs(save_path, exist_ok=True)

        plt.plot(history_aro.history['accuracy'], 'r', label='Arousal Accuracy ')
        plt.plot(history_val.history['accuracy'], 'b', label='Valence Accuracy')
        plt.title('Classification Accuracy Traning Phase')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(numpoints=2, fontsize=16)
        plt.savefig(save_path + 'Accuracy_{}_{}.png'.format(arch, ml))
        plt.close()

        plt.plot(history_aro.history['loss'], 'r', label='Arousal Loss')
        plt.plot(history_val.history['loss'], 'b', label='Valence Loss')
        plt.title(' Training Phase Loss ')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(numpoints=2, fontsize=16)
        plt.savefig(save_path + 'Loss_{}_{}.png'.format(arch, ml))
        plt.close()

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
        model[0].save(save_path + 'model_aro')
        model[1].save(save_path + 'model_val')


    # yhat_arousal = np.argmax(y_hat, axis=1)
    yhat_arousal = (y_hat[0] > 0.5).astype(int)
    yhat_valence = (y_hat[1] > 0.5).astype(int)

    # print(np.unique(yhat_arousal, return_counts=True))
    # yhat_valence = np.argmax(y_hat[1], axis=1)
    y_arousal = np.array(y[:, 0])
    y_valence = np.array(y[:, 1])


    print('yhat and y arousal in report', yhat_arousal.shape, y_arousal.shape)

    # print('report function, shapes are ', yhat_arousal.shape, yhat_valence.shape, y_arousal.shape, y_valence.shape)

    save_and_print_fn(y_arousal, yhat_arousal, 'arousal')
    save_and_print_fn(y_valence, yhat_valence, 'valence')

    plots(history[0], history[1])



def fed_report(y, y_hat, arch, ml, name):
    print(y.shape, y_hat.shape)
    y_ = np.squeeze(y)
    yhat_ = (y_hat > 0.5).astype(int)

    conf_mat = confusion_matrix(y_.tolist(), yhat_.tolist())
    report= classification_report(y_.tolist(), yhat_.tolist(), output_dict=True)
    df_ = pd.DataFrame.from_dict(report)


    print('\nReport of {}'.format(name))
    print(classification_report(y_.tolist(), yhat_.tolist()))
    print('\nConfusion Matrix for {}'.format(name))
    print(conf_mat)

    # Saving
    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

    wd = cur_path + deli + 'report' + deli

    if not os.path.isdir(wd):
      os.makedirs(wd, exist_ok=True)

    save_path = wd + date_time + deli
    os.makedirs(save_path, exist_ok=True)

    df_.to_csv(save_path+'rep_{}_{}_{}.csv'.format(name, arch, ml), index=False)
    # df_valence.to_csv(save_path+'rep_valence_{}_{}.csv'.format(arch, ml), index=False)
    print('Arousal and Valence report saved in the report directory with postfix date and time ,', date_time)
    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.to_csv(save_path + 'confusion_mat_{}_{}_{}.csv'.format(name, arch, ml), index=False)






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

