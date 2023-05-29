import numpy as np
from mne.filter import filter_data

from PluginCore.Decoder.module.CSP.base import FilterCSP,flatten,CSPBase

class FilterCSSP(FilterCSP):
    def __init__(self, sfreq, time_steps, window_start, window_length, csp_kwargs, T,
                 low_cut_hz=4,high_cut_hz=30,):
        super(FilterCSSP, self).__init__(sfreq=sfreq, time_steps=time_steps, window_start=window_start
                                         , window_length=window_length, csp_kwargs=csp_kwargs, low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)
        self.T = T

    def transform(self,X):
        x_filt = filter_data(X,sfreq=self.sfreq,l_freq=self.low_cut_hz,h_freq=self.high_cut_hz)
        x_filt = np.hstack([x_filt[..., self.T:], x_filt[..., :-self.T]])
        x_filt = x_filt[:, :, self.time_window[0]:self.time_window[1]]

        feature = self.csp.transform(x_filt)
        feature = flatten(feature)
        return feature

    def fit(self,X,y):
        x_filt = filter_data(X, sfreq=self.sfreq, l_freq=self.low_cut_hz, h_freq=self.high_cut_hz)
        x_filt = np.hstack([x_filt[..., self.T:], x_filt[..., :-self.T]])
        x_filt = x_filt[:, :, self.time_window[0]:self.time_window[1]]

        self.csp.fit(x_filt,y)

    def fit_transform(self,X,y):
        self.fit(X,y)
        feature = self.transform(X)
        return feature

class CSSPBase(CSPBase):
    def __init__(self, sfreq, time_steps, window_start, window_length, csp_kwargs, clf, T,
                 low_cut_hz=4,high_cut_hz=30):
        super(CSSPBase, self).__init__(sfreq=sfreq, time_steps=time_steps, window_start=window_start,
                                       window_length=window_length, csp_kwargs=csp_kwargs, clf=clf,
                                        low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz)
        self.filter_csp = FilterCSSP(sfreq=sfreq, time_steps=time_steps, window_start=window_start,
                                     window_length=window_length, csp_kwargs=csp_kwargs, T=T,
                                    low_cut_hz=low_cut_hz,high_cut_hz=high_cut_hz,)


