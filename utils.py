import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if 'posix' in os.name:
    deli = '/'
    cur_path = os.getcwd()
elif 'nt' in os.name:
    deli = '\\'
    cur_path = os.getcwd()



def report(y, y_hat, arch, ml):

    # Arousal

    y_, yhat = np.squeeze(y[:, 0]), np.squeeze(y_hat[0])
    yhat = (yhat > 0.5001).astype(int)
    conf_mat_arousal = confusion_matrix(y_.tolist(), yhat.tolist())
    report_arousal = classification_report(y_.tolist(), yhat.tolist(), output_dict=True)
    df_arousal = pd.DataFrame(report_arousal).transpose()

    print('\nReport of Arousal')
    print(classification_report(y_.tolist(), yhat.tolist()))
    print('\nConfusion Matrix')
    print(conf_mat_arousal)


    # Valence
    y_, yhat = np.squeeze(y[:, 1]), np.squeeze(y_hat[1])
    yhat = (yhat > 0.5001).astype(int)
    conf_mat_valence = confusion_matrix(y_.tolist(), yhat.tolist())
    report_valence = classification_report(y_.tolist(), yhat.tolist(), output_dict=True)
    df_valence = pd.DataFrame(report_valence).transpose()

    print('\nReport of Valence')
    print(classification_report(y_.tolist(), yhat.tolist()))
    print('\nConfusion Matrix')
    print(conf_mat_valence)

    # Saving
    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

    wd = cur_path + '{}report{}'.format(deli, deli)

    print('Arousal and Valence saved in the report directory with postfix date and time ,', date_time)
    if not os.path.isdir(wd):
      os.makedirs(wd, exist_ok=True)

    save_path = wd + date_time + deli
    os.makedirs(save_path, exist_ok=True)

    df_arousal.to_csv(save_path+'rep_arousal_{}_{}_{}.csv'.format(arch, ml), index=False)
    df_valence.to_csv(save_path+'rep_valence_{}_{}_{}.csv'.format(arch, ml), index=False)


def plots(history, arch, ml, name):
    wd = cur_path + '{}figs{}'.format(deli, deli)

    if not os.path.isdir(wd):
      os.makedirs(wd, exist_ok=True)

    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y__%H.%M.%S")

    save_path = wd + date_time + deli
    os.makedirs(save_path, exist_ok=True)

    plt.plot(history.history['loss'], 'm', label='Loss')
    plt.plot(history.history['arousal_loss'], 'r', label='Arousal Loss')
    plt.plot(history.history['valence_loss'], 'b', label='Valence Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(numpoints=1, fontsize=20)
    # plt.show()
    plt.savefig(save_path+'Loss_{}_{}_{}.png'.format(name, arch, ml))


    plt.plot(history.history['arousal_accuracy'], 'r', label='Arousal Accuracy')
    plt.plot(history.history['valence_accuracy'], 'b', label='Valence Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(numpoints=1, fontsize=20)
    plt.savefig(save_path+'Accuracy_model({})_{}_{}.png'.format(name, arch, ml))

    # lgnd = plt.legend(loc="lower left", numpoints=1, fontsize=20)
    # lgnd.legendHandles[0]._legmarker.set_markersize(20)


