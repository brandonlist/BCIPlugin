from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from copy import deepcopy
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def create_time_window(window_start,window_length,time_steps):
    """

    :param window_start:<int>
    :param window_length: <int>
    :param time_steps: <int>
    :return: <list>time_window
    """
    window_start,window_length = int(window_start),int(window_length)
    end = window_start+window_length if (window_start+window_length)<time_steps else time_steps
    return window_start,end

def flatten(X):
    return np.reshape(X,(X.shape[0],-1))

class FilterBank():
    def __init__(self,l_freq,h_freq,n_cuts):
        self.freqs = np.linspace(l_freq,h_freq,n_cuts+1)
        self.freqs_low = self.freqs[:n_cuts]
        self.freqs_high = self.freqs[1:]

        self.n_cuts = n_cuts
        self.l_freq = l_freq
        self.h_freq = h_freq

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.n_cuts:
            l_freq = self.freqs_low[self.idx]
            h_freq = self.freqs_high[self.idx]
            self.idx += 1
            return (l_freq,h_freq)
        else:
            raise StopIteration

class FilterCSP(BaseEstimator,ClassifierMixin):
    def __init__(self, sfreq, time_steps, window_start, window_length, csp_kwargs,
                 low_cut_hz=4,high_cut_hz=30):

        self.sfreq = sfreq
        self.time_steps = time_steps
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz

        self.window_start = window_start
        self.window_length = window_length

        self.time_window = create_time_window(window_start=self.window_start,
                                              window_length=self.window_length,
                                              time_steps=self.time_steps)

        self.csp = CSP(**csp_kwargs)

    def fit(self,X,y):
        bp_data = filter_data(X,sfreq=self.sfreq,l_freq=self.low_cut_hz,h_freq=self.high_cut_hz)
        bp_data = bp_data[:, :, self.time_window[0]:self.time_window[1]]

        self.csp.fit(bp_data,y)

    def transform(self,X):
        bp_data = filter_data(X,sfreq=self.sfreq,l_freq=self.low_cut_hz,h_freq=self.high_cut_hz)
        bp_data = bp_data[:, :, self.time_window[0]:self.time_window[1]]
        feature = self.csp.transform(bp_data)
        feature = flatten(feature)
        return feature

    def fit_transform(self,X,y):
        self.fit(X,y)
        feature = self.transform(X)
        return feature

class CSPBase(BaseEstimator,ClassifierMixin):
    def __init__(self, sfreq, time_steps, window_start, window_length, csp_kwargs, clf,
                 low_cut_hz=4,high_cut_hz=30):
        self.filter_csp = FilterCSP(sfreq=sfreq,time_steps=time_steps,window_start=window_start,window_length=window_length,
                                    csp_kwargs=csp_kwargs,low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)
        self.clf = deepcopy(clf)

    def fit(self,X,y):
        feature = self.filter_csp.fit_transform(X,y)
        self.clf.fit(feature,y)

    def transform(self,X):
        feature = self.filter_csp.transform(X)
        return feature

    def fit_transform(self,X,y):
        self.fit(X,y)
        feature = self.transform(X)
        return feature

    def predict(self,X):
        feature = self.transform(X)
        pred = self.clf.predict(feature)
        return pred

    def score(self,X,y,sample_weight=None):
        acc = accuracy_score(y,self.predict(X))
        return acc

    def logits(self,X):
        feature = self.transform(X)
        logits = self.clf.predict_proba(feature)
        return logits

class CSPBaseLDA(CSPBase):
    def __init__(self, sfreq, time_steps, window_start, window_length, csp_kwargs, shrinkage,
                 low_cut_hz=4,high_cut_hz=30):
        clf = LinearDiscriminantAnalysis(shrinkage=shrinkage)
        super(CSPBaseLDA, self).__init__(sfreq=sfreq,time_steps=time_steps,window_start=window_start,
                                         window_length=window_length,csp_kwargs=csp_kwargs,
                                         low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz,clf=clf)

class CSPBaseSVC(CSPBase):
    def __init__(self, sfreq, time_steps, window_start, window_length, csp_kwargs, C,
                 low_cut_hz=4,high_cut_hz=30):
        clf = SVC(C=C)
        super(CSPBaseSVC, self).__init__(sfreq=sfreq,time_steps=time_steps,window_start=window_start,
                                         window_length=window_length,csp_kwargs=csp_kwargs,
                                         low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz,clf=clf)

