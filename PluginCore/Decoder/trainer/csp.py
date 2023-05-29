import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import mne

from PluginCore.Decoder.ml import tsne_visualize
from PluginCore.Decoder.base import EEGDecoder


class CSPFactory():
    def __init__(self,Model):
        self.Model = Model

    def train(self, train_X, train_y, model, log_dir=None, save_name=None, max_epochs=None, verbose=True):
        try:
            train_X = train_X.numpy().astype(np.float)
            train_y = train_y.numpy()
        except:
            pass

        model.fit(train_X,train_y)
        if save_name is not None:
            with open(save_name, 'wb') as f:
                pickle.dump(model, f)
        return model

    def score(self, model, test_X, test_y):
        try:
            test_X = test_X.numpy().astype(np.float)
            test_y = test_y.numpy()
        except:
            pass

        return model.score(test_X,test_y)

    def predict(self, model, test_X):
        try:
            test_X = test_X.numpy().astype(np.float)
        except:
            pass

        return model.predict(test_X)

    def quick_acc(self, model, test_X, test_y, cuda=None):
        score = self.score(model, test_X, test_y)
        return score

    def train_finetune(self, train_X, train_y, little_X, little_y, model, log_dir=r'runs', save_name=None,
                       verbose=True):
        model_ori = self.train(train_X=train_X, train_y=train_y, model=model, log_dir=log_dir,
                               save_name=save_name, verbose=verbose)

        model_trans = deepcopy(model_ori)
        model_trans = self.train(train_X=little_X, train_y=little_y, model=model, log_dir=log_dir,
                                 save_name=save_name, verbose=verbose)
        return model_ori, model_trans

    def voting(self, models, n_classes, X, weights=None):
        if weights is None:
            weights = [1 for i in range(len(models))]

        preds = []
        for i_model, model in enumerate(models):
            preds.append(self.predict(model,X))

        pred_out = torch.zeros([X.shape[0], n_classes])
        for j, pred in enumerate(preds):
            for i, _ in enumerate(pred_out):
                pred_out[i][pred[i]] += weights[j]

        pred_out = torch.argmax(pred_out, dim=1)
        return pred_out

class CSPRandomSearch():
    def __init__(self, Model, **kwargs):
        if 'n_iter' in kwargs.keys():
            self.n_iter = kwargs['n_iter']
        else:
            self.n_iter = 4

    def search(self, params, model, verbose=False):
        gs = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=self.n_iter, refit=False, cv=3, scoring='accuracy', verbose=4)
        return gs



class CSPFilter():
    def __init__(self, filter_func, model):
        self.filter_func = filter_func
        self.model = model

    def compile(self, model):
        self.model = model

    def visualize_train_test_dist(self, train_X, train_y, test_X, test_y):
        classes_train = np.unique(train_y)
        legends = ['train_' + str(c) for c in classes_train]

        train_X = train_X.numpy().astype(np.float)
        test_X = test_X.numpy().astype(np.float)

        train_f = self.model.transform(train_X)
        test_f = self.model.transform(test_X)

        BigNum = 100
        test_y += BigNum
        classes_test = np.unique(test_y)
        legends.extend(['test_' + str(c-BigNum) for c in classes_test])

        all_classes = np.concatenate([classes_train, classes_test])
        all_feature = np.concatenate([train_f, test_f])
        all_y = np.concatenate([train_y, test_y])

        idxs = [all_y==i for i in all_classes]
        tsne_visualize(all_feature,idxs,legends)

    def track_time(self, all_X, all_y, n_inter):
        time_idxs = np.linspace(0, len(all_X), n_inter + 1, dtype=np.int)
        time_code = list(range(n_inter))
        y_idxs = np.zeros_like(all_y)
        for i, _ in enumerate(time_idxs[:-1]):
            y_idxs[time_idxs[i]:time_idxs[i + 1]] = time_code[i]

        all_feature = self.model.transform(all_X.numpy().astype(np.float))
        tsne_visualize(all_feature, [np.ones(all_feature.shape[0])==1], legends=None, colors=y_idxs)

class CSPWrapper():
    def __init__(self,train_factory, inspector_factory):
        self.train_factory = train_factory
        self.inspector_factory = inspector_factory

    def compile(self, model, sub_channel, time_interval):
        sub_model = deepcopy(model)
        sub_model.filter_csp.time_window = time_interval
        sub_model.filter_csp.window_length = time_interval[1] - time_interval[0]
        sub_model.filter_csp.window_start = time_interval[0]
        return sub_model

    def train(self, model, train_X, train_y, channels, time_interval):
        train_X = train_X[:,channels,time_interval[0]:time_interval[1]]

        ans = self.train_factory.train(train_X=train_X,train_y=train_y,model=model)
        return ans

    def inspect(self, test_X, test_y, model, channels, time_interval):
        test_X = test_X[:, channels, time_interval[0]:time_interval[1]]

        re = self.inspector_factory.inspect(test_X, test_y, model)
        return re

    def plot_metric(self, res, metric):
        #TODO:make time interval actually following time sequence
        re_wrappers = [re[0] for re in res]
        time_interval_wrappers = [re[2] for re in res]

        plt.plot([str(t) for t in time_interval_wrappers], [r[metric] for r in re_wrappers])

    def draw_topomap(self, res, info):
        re_wrappers = [re[0] for re in res]

        spat_acc = [r['acc'] for r in re_wrappers]
        spat_acc = np.array(spat_acc)
        pos = np.array([(c['loc'][0], c['loc'][1]) for c in info['chs']])

        mne.viz.plot_topomap(data=spat_acc, pos=pos,
                             show_names=True, names=info['ch_names'])
