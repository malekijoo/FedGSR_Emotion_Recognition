import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

import kmeans1d
import neurokit2 as nk
import scipy.signal as signal
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)




class CASE():

    def __init__(self, phy_dir, ann_dir, arch='CENT'):

        print('\nBuilding the DATASET. ')
        self.phy_add = CASE.read(phy_dir, '*.csv')
        self.ann_add = CASE.read(ann_dir, '*.csv')
        self.NumberOfUsers = len(self.phy_add)
        self.arch = arch

        self.x, self.y, self.cwt, self.sf, self.ss, self.resp = [], [], [], [], [], []

        if 'posix' in os.name:
            self.deli = '/'
            self.cur_path = os.getcwd()
        elif 'nt' in os.name:
            self.deli = '\\'
            self.cur_path = os.getcwd()


    @staticmethod
    def read(pathdir, f_extention):
        return np.array(tf.io.gfile.glob(pathdir + f_extention))

    @staticmethod
    def read_dataframe(address, columns):
        return pd.read_csv(address, usecols=columns, index_col=False)

    @staticmethod
    def kmeans(signal, k=2):
        labels, centroids = kmeans1d.cluster(signal, k=k)
        return labels

    @staticmethod
    def inc_dec_map(label):
        """Mapping function helps us map the label to several predifined classes"""

        valence = label['valence'].diff().mean()
        arousal = label['arousal'].diff().mean()

        if arousal >= 0 and valence >= 0:  # HIGH arousal - HIGH valence (HH)
            return [1, 1]

        elif arousal >= 0 and valence < 0:  # HIGH arousal - LOW valence (HL)
            return [1, 0]

        elif arousal < 0 and valence >= 0:  # LOW arousal - HIGH valence (LH)
            return [0, 1]

        elif arousal < 0 and valence < 0:  # LOW arousal - LOW valence (LL)
            return [0, 0]

    @staticmethod
    def mean_interval_mapping(label):
        """Mapping function helps us map the label to several predifined classes"""

        valence = label['valence'].mean()
        arousal = label['arousal'].mean()

        if arousal >= 5 and valence >= 5:  # HIGH arousal - HIGH valence (HH)
            return [1, 1]

        elif arousal >= 5 and valence < 5:  # HIGH arousal - LOW valence (HL)
            return [1, 0]

        elif arousal < 5 and valence >= 5:  # LOW arousal - HIGH valence (LH)
            return [0, 1]

        elif arousal < 5 and valence < 5:  # LOW arousal - LOW valence (LL)
            return [0, 0]

    @staticmethod
    def kmeans_mapping(label):
        valence = label['k_valence'].mean()
        arousal = label['k_arousal'].mean()
        # print('valence and arousal mean() ', valence, int(round(valence)), arousal, int(round(arousal)))
        return [int(round(arousal)), int(round(valence))]

    @staticmethod
    def round_interval_mean(label):
        return [int(round(label['arousal'].mean())), int(round(label['valence'].mean()))]


    @staticmethod
    def spectral_statics(sample):
        """
        Compute the spectral mean (first spectral moment)
        Compute the spectral variance (second spectral moment)
        Compute the spectral skewness (third spectral moment),
        Compute the spectral kurtosis (fourth spectral moment),

        """
        return [(np.sum(abs(sample)) / len(sample)), np.var(abs(sample)),
                stats.skew(abs(sample)), stats.kurtosis(abs(sample))]

    @staticmethod
    def minmax__norm(raw_signal):
        return (raw_signal - raw_signal.min()) / (raw_signal.max() - raw_signal.min())


    @staticmethod
    def zscore__norm(raw_signal):
        return (raw_signal - raw_signal.mean()) / (raw_signal.std())

    @staticmethod
    def decompose(raws_signal, only_phasic=True):
        sig, info = nk.eda_process(raws_signal, sampling_rate=1000)
        if only_phasic:
            return sig['EDA_Phasic']
        else:
            return sig

    @staticmethod
    def cwt(sample):
        widths = np.arange(1, 8)
        return signal.cwt(sample, signal.ricker, widths=widths).T  # shape (20, len(sample)])

    @staticmethod
    def spectral_flux(sample):
        # convert to frequency domain
        f, t, spectrum = CASE.stft(sample)
        timebins, freqbins = np.shape(spectrum)
        return np.sqrt(np.sum(np.diff(np.abs(spectrum)) ** 2, axis=1)) / freqbins

    @staticmethod
    def stft(sample):
        fs = 10e3
        f, t, z = signal.stft(sample, fs, nperseg=100)
        return f, t, z



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

    #                       Session Chunking

    ###############################################################################################################
    ###############################################################################################################

    @staticmethod
    def session_chunk(gp_sess_ann_df, gp_sess_phy_df):

        interval_length = 20


        temp_x, temp_y, temp_cwt, temp_sf, temp_ss, temp_resp = [], [], [], [], [], []
        str_p = gp_sess_ann_df['jstime'].iloc[0]

        for k in range(interval_length, gp_sess_ann_df.shape[0], interval_length):

            end_p = gp_sess_ann_df['jstime'].iloc[k]
            label = gp_sess_ann_df[(gp_sess_ann_df['jstime'] <= end_p) & (gp_sess_ann_df['jstime'] > str_p)]

            # clss = CASE.round_interval_mean(label)
            # clss = CASE.inc_dec_map(label)
            clss = CASE.mean_interval_mapping(label)
            # clss = CASE.kmeans_mapping(label)
            # print('class', clss)

            sample_df = gp_sess_phy_df[(gp_sess_phy_df['daqtime'] <= end_p) & (gp_sess_phy_df['daqtime'] > str_p)]
            sample = sample_df['gsr_phasic']



            cwt = CASE.cwt(sample)
            # stft = CASE.extract_stft()
            sf = CASE.spectral_flux(sample)
            ss = CASE.spectral_statics(sample)
            resp = [sample_df['peaks'].fillna(0).tolist(), sample_df['risetime'].fillna(0).tolist(),
                    sample_df['height'].fillna(0).tolist(), sample_df['recovery'].fillna(0).tolist()]


            if sample.shape[0] == (interval_length*50): # interval_length * 50(each label cover 50 value in sample data)
                temp_x.append(sample)
                temp_y.append(np.array(clss))
                temp_cwt.append(np.array(cwt))
                temp_sf.append(np.array(sf))
                temp_ss.append(np.array(ss))
                temp_resp.append(np.array(resp).T)


            str_p = gp_sess_ann_df['jstime'].iloc[k]


        return np.array(temp_x), np.array(temp_y), np.array(temp_cwt), np.array(temp_sf), np.array(temp_ss), np.array(temp_resp),

    ###############################################################################################################
    ###############################################################################################################

    #                       PreProcess

    ###############################################################################################################
    ###############################################################################################################

    def preprocess(self, usr):

        phy_col = ['daqtime', 'gsr', 'video']
        ann_col = ['jstime', 'valence', 'arousal', 'video']


        username = self.phy_add[usr].split(self.deli)[-1].split('.')[0]
        tusrname = self.ann_add[usr].split(self.deli)[-1].split('.')[0]

        assert username == tusrname

        self.phy_df = CASE.read_dataframe(self.phy_add[usr], phy_col)
        self.ann_df = CASE.read_dataframe(self.ann_add[usr], ann_col)

        # normalization and decomposition are applied on each person's GSR signal
        # self.phy_df['gsr'] = CASE.minmax__norm(self.phy_df['gsr'])
        self.phy_df['gsr'] = CASE.zscore__norm(self.phy_df['gsr'])

        temp_sig = CASE.decompose(self.phy_df['gsr'], only_phasic=False)

        self.phy_df['gsr_phasic'] = temp_sig['EDA_Phasic']
        self.phy_df['peaks'] = temp_sig['SCR_Peaks']
        self.phy_df['risetime'] = temp_sig['SCR_RiseTime']
        self.phy_df['height'] = temp_sig['SCR_Height']
        self.phy_df['recovery'] = temp_sig['SCR_Recovery']

        # self.ann_df['valence'] = CASE.zscore__norm(self.ann_df['valence'])
        # self.ann_df['arousal'] = CASE.zscore__norm(self.ann_df['arousal'])

        # self.ann_df['valence'] = CASE.kmeans(self.ann_df['valence'], 2)
        # self.ann_df['arousal'] = CASE.kmeans(self.ann_df['arousal'], 2)

        # print('arousal ', self.ann_df['arousal'].value_counts())
        # print('valence ', self.ann_df['valence'].value_counts())
        gp_phy_df = self.grouped_by_video_id(self.phy_df, feature='video')
        gp_ann_df = self.grouped_by_video_id(self.ann_df, feature='video')

        return gp_phy_df, gp_ann_df

    ###############################################################################################################
    ###############################################################################################################

    #                       CENT Process

    ###############################################################################################################
    ###############################################################################################################

    def cent_process(self):
        for usr in range(self.NumberOfUsers):
            # for usr in range(3):
            print('usr : ', usr)
            gp_phy_df, gp_ann_df = self.preprocess(usr=usr)

            for sess in range(gp_ann_df.shape[0]):
                gp_sess_ann_df = gp_ann_df[sess]
                gp_sess_phy_df = gp_phy_df[sess]
                # gp_sess_phy_df = phy.get_group(ann[sess][0])
                x, y, cwt, sf, ss, resp = CASE.session_chunk(gp_sess_ann_df=gp_sess_ann_df, gp_sess_phy_df=gp_sess_phy_df)
                self.x.append(x), self.y.append(y), self.cwt.append(cwt)
                self.sf.append(sf), self.ss.append(ss), self.resp.append(resp)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.cwt = np.concatenate(self.cwt)
        self.sf = np.concatenate(self.sf)
        self.ss = np.concatenate(self.ss)
        self.resp = np.concatenate(self.resp)

    ###############################################################################################################
    ###############################################################################################################

    #                       FED Process

    ###############################################################################################################
    ###############################################################################################################


    def fed_process(self):
        print('making the DATASET federated')

        for usr in range(self.NumberOfUsers):
            # for usr in range(2):

            gp_phy_df, gp_ann_df = self.preprocess(usr=usr)
            sess_x, sess_y, sess_cwt, sess_sf, sess_ss, sess_resp = [], [], [], [], [], []

            for sess in range(gp_ann_df.shape[0]):
                gp_sess_ann_df = gp_ann_df[sess]
                gp_sess_phy_df = gp_phy_df[sess]
                # gp_sess_phy_df = phy.get_group(ann[sess][0])
                x, y, cwt, sf, ss, resp = CASE.session_chunk(gp_sess_ann_df=gp_sess_ann_df, gp_sess_phy_df=gp_sess_phy_df)
                sess_x.append(x), sess_y.append(y), sess_cwt.append(cwt)
                sess_sf.append(sf), sess_ss.append(ss), sess_resp.append(resp)

            self.x.append(np.array(sess_x))
            self.y.append(np.array(sess_y))
            self.cwt.append(np.array(sess_cwt))
            self.sf.append(np.array(sess_sf))
            self.ss.append(np.array(sess_ss))
            self.resp.append(np.array(sess_resp))


        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.cwt = np.array(self.cwt)
        self.sf = np.array(self.sf)
        self.ss = np.array(self.ss)
        self.resp = np.array(self.resp)


    def save_dataset(self):
        start_time = datetime.now()
        wd = self.cur_path + '{}dataset{}'.format(self.deli, self.deli)

        if self.arch == 'CENT':
            print('\nCentralized Architecture DATASET')
            if not os.path.isdir(wd+self.arch+self.deli):
                os.makedirs(wd+self.arch+self.deli, exist_ok=True)
            self.cent_process()
            print('building DATASET time ==', datetime.now()-start_time)

        elif self.arch == 'FED':
            print('\nFederated Architecture DATASET')
            if not os.path.isdir(wd+self.arch+self.deli):
                os.makedirs(wd+self.arch+self.deli, exist_ok=True)
            self.fed_process()
            print('building DATASET time ', datetime.now()-start_time)

            # self.fed_process()

        # print(self.x.shape, self.x.shape, self.cwt.shape)
        np.save(wd+self.arch+self.deli+'x.npy', np.array(self.x))  # save
        np.save(wd+self.arch+self.deli+'y.npy', np.array(self.y))  # save
        np.save(wd+self.arch+self.deli+'cwt.npy', np.array(self.cwt))  # save
        # print(self.sf.shape, self.ss.shape, self.resp.shape)
        np.save(wd+self.arch+self.deli+'sf.npy', np.array(self.sf))  # save
        np.save(wd+self.arch+self.deli+'ss.npy', np.array(self.ss))  # save
        np.save(wd+self.arch+self.deli+'resp.npy', np.array(self.resp))  # save

        # print('dataset is saved in "x.npy", "y.npy", and cwt.npy files')

    def load_data(self, x='/x.npy', y='/y.npy', cwt='/cwt.npy'):

        print('Loading the Dataset. ')
        wd = self.cur_path + '{}dataset{}'.format(self.deli, self.deli)

        if not os.path.isdir(wd):
            os.makedirs(wd, exist_ok=True)
        if not os.path.isfile(wd+self.arch+self.deli+'y.npy'):
            self.save_dataset()


        x = np.load(wd+self.arch+self.deli+'x.npy', allow_pickle=True)
        y = np.load(wd+self.arch+self.deli+'y.npy', allow_pickle=True)
        cwt = np.load(wd+self.arch+self.deli+'cwt.npy', allow_pickle=True)
        sf = np.load(wd+self.arch+self.deli+'sf.npy', allow_pickle=True)
        ss = np.load(wd+self.arch+self.deli+'ss.npy', allow_pickle=True)
        resp = np.load(wd+self.arch+self.deli+'resp.npy', allow_pickle=True)



        return x, y, cwt, sf, ss, resp

