from sklearn.feature_selection import RFE

from PluginCore.Decoder.module.CSP.FBCSP import FilterBankCSP
from PluginCore.Decoder.module.CSP.base import CSPBase

class SBCSP(CSPBase):
    def __init__(self, sfreq, time_steps, window_start, window_length, clf,
                 l_freq,h_freq,n_cuts, csp_kwargs={'n_components':4}):
        super(SBCSP, self).__init__(sfreq=sfreq, time_steps=time_steps, window_start=window_start,
                                       window_length=window_length, csp_kwargs=csp_kwargs, clf=clf,
                                       low_cut_hz=l_freq, high_cut_hz=h_freq)
        self.filter_csp = FilterBankCSP(sfreq=sfreq, time_steps=time_steps, window_start=window_start,
                                        window_length=window_length, select_ratio=1,
                                        l_freq=l_freq, h_freq=h_freq, n_cuts=n_cuts, csp_kwargs=csp_kwargs)
        self.rfe = RFE(estimator=self.clf,step=1)


    def fit(self,X,y):
        feature = self.filter_csp.fit_transform(X,y)
        self.rfe.fit(feature,y)
        feature = self.rfe.transform(feature)
        self.clf.fit(feature,y)

    def transform(self,X):
        feature = self.filter_csp.transform(X)
        feature = self.rfe.transform(feature)
        return feature

