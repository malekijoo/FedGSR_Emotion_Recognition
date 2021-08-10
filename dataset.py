import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf


import neurokit2 as nk
import scipy.signal as signal

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)




class CASE:

    def __init__(self, phy_dir, ann_dir, arch='CENT'):
        print('\nBuilding the DATASET. ')

        self.phy_add = CASE.read(phy_dir, '*.csv')
        self.ann_add = CASE.read(ann_dir, '*.csv')
        self.NumberOfUsers = len(self.phy_add)
        self.arch = arch
        self.x, self.y, self.cwt = [], [], []


    @staticmethod
    def read(pathdir, f_extention):
        return np.array(tf.io.gfile.glob(str(pathdir) + f_extention))

    @staticmethod
    def dataframe(address, columns):
        return pd.read_csv(address, usecols=columns, index_col=False)

    #
    @staticmethod
    def mapping(row):
        """Mapping function helps us map the label to several predifined classes"""

        if row['arousal'] >= 0 and row["valence"] >= 0:  # HIGH arousal - HIGH valence (HH)
            return [1, 1]

        elif row['arousal'] >= 0 and row["valence"] < 0:  # HIGH arousal - LOW valence (HL)
            return [1, 0]

        elif row['arousal'] < 0 and row["valence"] >= 0:  # LOW arousal - HIGH valence (LH)
            return [0, 1]

        elif row['arousal'] < 0 and row["valence"] < 0:  # LOW arousal - LOW valence (LL)
            return [0, 0]

    @staticmethod
    def minmax__norm(raw_signal):
        return ((raw_signal - raw_signal.min()) / (raw_signal.max() - raw_signal.min()))

    @staticmethod
    def decompose(raws_signal):
        sig, info = nk.eda_process(raws_signal, sampling_rate=1000)
        return sig['EDA_Phasic']
        # sig['EDA_Phasic'] sig['EDA_Tonic'] sig['SCR_Peaks'] sig['SCR_RiseTime']

    @staticmethod
    def feature_extraction(sample):
        widths = np.arange(1, 21)
        return signal.cwt(sample['gsr_phasic'], signal.ricker, widths=widths).T  # shape (20, 500)


    @staticmethod
    def grouped_by_video_id(df, feature='video'):
        grouped_by_video_id = df.groupby(feature)
        df_list_grouped_by_video_id = []
        for grouped_df in grouped_by_video_id:
            # print('gp by type ', type(grouped_df[1))
            df_list_grouped_by_video_id.append(grouped_df[1])
        # a list of tuples including (video id, data frame grouped by video id)
        return np.array(df_list_grouped_by_video_id) # it returns a numpy array of pandas data frames grouped by 'VIDEO ID


    ###############################################################################################################
    ###############################################################################################################

    #                       CENT

    ###############################################################################################################
    ###############################################################################################################

    @staticmethod
    def session_chunk(gp_sess_ann_df, gp_sess_phy_df):
        temp_x, temp_y, temp_cwt = [], [], []
        str_p = gp_sess_ann_df['jstime'].iloc[0]
        #
        for k in range(20, gp_sess_ann_df.shape[0], 20):
            end_p = gp_sess_ann_df['jstime'].iloc[k]
            label = gp_sess_ann_df[(gp_sess_ann_df['jstime'] <= end_p) & (gp_sess_ann_df['jstime'] > str_p)]
            valence = label['valence'].diff().mean()
            arousal = label['arousal'].diff().mean()
            sample = gp_sess_phy_df[(gp_sess_phy_df['daqtime'] <= end_p) & (gp_sess_phy_df['daqtime'] > str_p)]

            clss = CASE.mapping(row={'arousal': arousal, 'valence': valence})
            cwt = CASE.feature_extraction(sample=sample)
            str_p = gp_sess_ann_df['jstime'].iloc[k]

            if sample.shape[0] == 1000:
                temp_x.append(sample['gsr_phasic'].to_numpy())
                temp_y.append(np.array(clss))
                temp_cwt.append(cwt)

        return np.array(temp_x), np.array(temp_y), np.array(temp_cwt)


    def process(self, usr):

        phy_col = ['daqtime', 'gsr', 'video']
        ann_col = ['jstime', 'valence', 'arousal', 'video']

        username = self.phy_add[usr].split('/')[-1].split('.')[0]
        tusrname = self.ann_add[usr].split('/')[-1].split('.')[0]
        assert username == tusrname

        self.phy_df = CASE.dataframe(self.phy_add[usr], phy_col)
        self.ann_df = CASE.dataframe(self.ann_add[usr], ann_col)

        # min max normalization and decomposition are applied on each person's GSR signal
        self.phy_df['gsr'] = CASE.minmax__norm(self.phy_df['gsr'])
        self.phy_df['gsr_phasic'] = CASE.decompose(self.phy_df['gsr'])

        gp_phy_df = self.grouped_by_video_id(self.phy_df, feature='video')
        gp_ann_df = self.grouped_by_video_id(self.ann_df, feature='video')

        return gp_phy_df, gp_ann_df


    def cent_process(self):

        for usr in range(self.NumberOfUsers):
            # for i in range(5):

            gp_phy_df, gp_ann_df = self.process(usr=usr)

            for sess in range(gp_ann_df.shape[0]):
                gp_sess_ann_df = gp_ann_df[sess]
                gp_sess_phy_df = gp_phy_df[sess]
                # gp_sess_phy_df = phy.get_group(ann[sess][0])
                x, y, cwt = CASE.session_chunk(gp_sess_ann_df=gp_sess_ann_df, gp_sess_phy_df=gp_sess_phy_df)
                self.x.append(x), self.y.append(y), self.cwt.append(cwt)

            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.cwt = np.array(self.cwt)

    ###############################################################################################################
    ###############################################################################################################

    #                       FED

    ###############################################################################################################
    ###############################################################################################################


    def fed_process(self):
        print('making the DATASET federated')

        for usr in range(self.NumberOfUsers):
        # for usr in range(10):

            gp_phy_df, gp_ann_df = self.process(usr=usr)
            sess_x, sess_y, sess_cwt = [], [], []

            for sess in range(gp_ann_df.shape[0]):
                gp_sess_ann_df = gp_ann_df[sess]
                gp_sess_phy_df = gp_phy_df[sess]
                # gp_sess_phy_df = phy.get_group(ann[sess][0])
                x, y, cwt = CASE.session_chunk(gp_sess_ann_df=gp_sess_ann_df, gp_sess_phy_df=gp_sess_phy_df)
                sess_x.append(x), sess_y.append(y), sess_cwt.append(cwt)

            self.x.append(np.array(sess_x))
            self.y.append(np.array(sess_y))
            self.cwt.append(np.array(sess_cwt))

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.cwt = np.array(self.cwt)



    def save_dataset(self):
        start_time = datetime.now()
        if self.arch == 'CENT':
            print('\n Centralized Architecture DATASET')
            if not os.path.isdir('./dataset/CENT/'):
                os.makedirs('./dataset/CENT/', exist_ok=True)
            self.cent_process()
            print('building DATASET duration time ', datetime.now()-start_time)

        elif self.arch == 'FED':
            print('\n Federated Architecture DATASET')
            if not os.path.isdir('./dataset/FED/'):
                os.makedirs('./dataset/FED/', exist_ok=True)
            self.fed_process()
            print('building DATASET duration time ', datetime.now()-start_time)

            # self.fed_process()

        print(self.x.shape, self.x.shape, self.cwt.shape)
        np.save('./dataset/{}/x.npy'.format(self.arch), np.array(self.x))  # save
        np.save('./dataset/{}/y.npy'.format(self.arch), np.array(self.y))  # save
        np.save('./dataset/{}/cwt.npy'.format(self.arch), np.array(self.cwt))  # save

        # print('dataset is saved in "x.npy", "y.npy", and cwt.npy files')

    def load_data(self, x='/x.npy', y='/y.npy', cwt='/cwt.npy'):
        print('Loading the Dataset. ')
        wd = './dataset/'
        if not os.path.isdir(wd):
            os.makedirs(wd, exist_ok=True)
        if not os.path.isfile(wd+'{}/y.npy'.format(self.arch)):
            self.save_dataset()


        x = np.load('./dataset/{}/x.npy'.format(self.arch), allow_pickle=True)
        y = np.load('./dataset/{}/y.npy'.format(self.arch), allow_pickle=True)
        cwt = np.load('./dataset/{}/cwt.npy'.format(self.arch), allow_pickle=True)

        if self.arch == 'CENT':
            x = np.concatenate(x, np.load('./dataset/CENT/x1.npy', allow_pickle=True), axis=0)
            y = np.concatenate(y, np.load('./dataset/CENT/y1.npy', allow_pickle=True), axis=0)
            cwt = np.concatenate(cwt, np.load('./dataset/CENT/cwt1.npy', allow_pickle=True), axis=0)


        return x, y, cwt

