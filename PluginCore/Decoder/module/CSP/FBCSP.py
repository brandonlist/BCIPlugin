import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from copy import deepcopy

from PluginCore.Decoder.module.CSP.base import FilterBank,create_time_window,flatten,CSPBase

class FilterBankCSP():
    def __init__(self, sfreq, time_steps, window_start, window_length,
                 select_ratio,l_freq,h_freq,n_cuts, csp_kwargs={'n_components':4}):
        self.sfreq = sfreq
        self.time_steps = time_steps

        self.window_start = window_start
        self.window_length = window_length

        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)

        self.n_cuts = n_cuts
        self.n_components = csp_kwargs['n_components']

        self.csps = []
        for i in range(n_cuts):
            self.csps.append(CSP(**csp_kwargs))

        self.select_ratio = select_ratio
        self.filter_bank = FilterBank(l_freq=l_freq,h_freq=h_freq,n_cuts=n_cuts)

        self.k = int(n_cuts * self.n_components * select_ratio)
        self.selector = SelectKBest(mutual_info_classif, k=self.k)

    def fit(self,X,y):
        features = np.zeros((X.shape[0],self.n_cuts,self.n_components))
        for i,(l_freq,h_freq) in enumerate(self.filter_bank):
            bp_data = filter_data(X,sfreq=self.sfreq,l_freq=l_freq,h_freq=h_freq)
            bp_data = bp_data[:, :, self.time_window[0]:self.time_window[1]]

            self.csps[i].fit(bp_data,y)
            feature = self.csps[i].transform(bp_data)
            features[:, i, :] = feature
        features = flatten(features)
        self.selector.fit(features,y)

    def transform(self,X):
        features = np.zeros((X.shape[0],self.n_cuts,self.n_components))
        for i,(l_freq,h_freq) in enumerate(self.filter_bank):
            bp_data = filter_data(X,sfreq=self.sfreq,l_freq=l_freq,h_freq=h_freq)
            bp_data = bp_data[:, :, self.time_window[0]:self.time_window[1]]

            feature = self.csps[i].transform(bp_data)
            features[:,i,:] = feature
        features = flatten(features)
        features = self.selector.transform(features)
        return features

    def fit_transform(self,X,y):
        self.fit(X,y)
        feature = self.transform(X)
        return feature


class FBCSP(CSPBase):
    def __init__(self, sfreq, time_steps, window_start, window_length, clf,
                 select_ratio,l_freq,h_freq,n_cuts, csp_kwargs={'n_components':4}):
        super(FBCSP, self).__init__(sfreq=sfreq, time_steps=time_steps, window_start=window_start,
                                       window_length=window_length, csp_kwargs=csp_kwargs, clf=clf,
                                       low_cut_hz=l_freq, high_cut_hz=h_freq)

        self.filter_csp = FilterBankCSP(sfreq=sfreq, time_steps=time_steps, window_start=window_start,
                                             window_length=window_length, select_ratio=select_ratio,
                                             l_freq=l_freq,h_freq=h_freq,n_cuts=n_cuts, csp_kwargs=csp_kwargs)


class FBCSPSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, sfreq, time_steps, window_start, window_length,
                 select_ratio,l_freq,h_freq,n_cuts):
        super(FBCSPSearch, self).__init__()
        self.clf = deepcopy(SVC(probability=True))
        self.csp_kwargs = {'n_components': 4}

        self.sfreq = sfreq
        self.time_steps = time_steps
        self.window_start = window_start
        self.window_length = window_length
        self.select_ratio = select_ratio
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.n_cuts = n_cuts

        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)
        self.csps = []
        for i in range(n_cuts):
            self.csps.append(CSP(**self.csp_kwargs))

        self.filter_bank = FilterBank(l_freq=l_freq,h_freq=h_freq,n_cuts=n_cuts)
        # self.selector = SelectKBest(mutual_info_classif, k=self.k)
        self.mask = None

    def fit(self, X, y):
        X = X.numpy().astype(np.float)
        y = y.numpy()

        features = np.zeros((X.shape[0], self.n_cuts, self.csp_kwargs['n_components']))
        for i, (l_freq, h_freq) in enumerate(self.filter_bank):
            bp_data = filter_data(X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq)
            bp_data = bp_data[:, :, self.time_window[0]:self.time_window[1]]

            self.csps[i].fit(bp_data, y)
            feature = self.csps[i].transform(bp_data)
            features[:, i, :] = feature
        features = flatten(features)
        k = int(self.n_cuts * self.csp_kwargs['n_components'] * self.select_ratio)

        filter = SelectKBest(mutual_info_classif, k)
        filter.fit(features, y)
        self.mask = filter._get_support_mask()
        features = filter.transform(features)

        self.clf.fit(features, y)

    def transform(self,X):
        X = X.numpy().astype(np.float)

        features = np.zeros((X.shape[0], self.n_cuts, self.csp_kwargs['n_components']))
        for i, (l_freq, h_freq) in enumerate(self.filter_bank):
            bp_data = filter_data(X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq)
            bp_data = bp_data[:, :, self.time_window[0]:self.time_window[1]]

            feature = self.csps[i].transform(bp_data)
            features[:, i, :] = feature
        features = flatten(features)
        features = features[:,self.mask]

        return features

    def fit_transform(self,X,y):
        self.fit(X, y)
        feature = self.transform(X)
        return feature

    def predict(self,X):
        feature = self.transform(X)
        pred = self.clf.predict(feature)
        return pred

    def score(self, X, y, sample_weight=None):
        X = X.numpy().astype(np.float)
        y = y.numpy()

        acc = accuracy_score(y, self.predict(X))
        return acc

    def logits(self,X):
        feature = self.transform(X)
        logits = self.clf.predict_proba(feature)
        return logits




