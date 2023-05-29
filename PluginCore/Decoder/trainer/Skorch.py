from skorch.callbacks import EarlyStopping,GradientNormClipping,LRScheduler,TensorBoard
from torch.optim.lr_scheduler import *
from skorch.classifier import NeuralNetClassifier
import tensorboardX
import torch
import pickle
from copy import deepcopy
from torch import nn
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

class SkorchFactory():
    def __init__(self,Model,**kwargs):
        self.Model = Model

        self.gradient_norm_clipping = GradientNormClipping(gradient_clip_value=1e-3)
        self.cosanllr = LRScheduler(policy=CosineAnnealingLR, T_max=20)
        # self.writer = tensorboardX.SummaryWriter(logdir='runs')
        # self.tensorboard_cb = TensorBoard(writer=self.writer)     #import pickle.dump error
        self.criterion = torch.nn.CrossEntropyLoss
        self.optimizer = torch.optim.AdamW
        self.weight_decay = 0.9
        self.cuda = 'cuda'
        self.patience = 1000

        if 'lr' in kwargs.keys():
            self.lr = kwargs['lr']
        else:
            self.lr = 6e-4

        if 'max_epochs' in kwargs.keys():
            self.max_epochs = kwargs['max_epochs']
        else:
            self.max_epochs = 100

        if 'es_patience' in kwargs.keys():
            self.patience = kwargs['es_patience']


        self.early_stopping = EarlyStopping(monitor='valid_loss', patience=self.patience, threshold=1e-5)

    def compile(self, model):
        Trainner = NeuralNetClassifier(
            model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            device=self.cuda,
            max_epochs=1,
            callbacks=[
                # self.tensorboard_cb,
                self.early_stopping,
                self.gradient_norm_clipping,
                ('lr_scheduler', self.cosanllr)
            ],
        )
        return Trainner

    def train(self, train_X, train_y, model,  log_dir=r'runs', save_name=None, verbose=True):
        Trainner = self.compile(model)

        Trainner.warm_start = True
        Trainner.max_epochs = self.max_epochs
        Trainner.verbose = verbose

        Trainner.fit(train_X,train_y)

        if save_name is not None:
            with open(save_name, 'wb') as f:
                pickle.dump(Trainner, f)

        return Trainner

    def score(self, model, test_X, test_y):
        return model.score(test_X, test_y)

    def quick_acc(self, model, test_X, test_y, cuda=True):
        """

        :param model:  NeuralNetClassifier,
        :param test_X:
        :param test_y:
        :param cuda:
        :return:
        """
        module = model.module
        if cuda:
            return torch.sum(torch.argmax(module(test_X.cuda()), dim=1) == test_y.cuda()) / len(test_y)
        else:
            return torch.sum(torch.argmax(module(test_X), dim=1) == test_y) / len(test_y)

    def train_finetune(self, train_X, train_y, little_X, little_y,  model, log_dir=r'runs', save_name=None, verbose=True):
        model_ori = self.train(train_X=train_X, train_y=train_y, model=model, log_dir=log_dir,
                               save_name=save_name, verbose=verbose)

        model_trans = deepcopy(model_ori)
        model_trans = self.train(train_X=little_X, train_y=little_y, model=model_trans.module, log_dir=log_dir,
                               save_name=save_name, verbose=verbose)
        return model_ori, model_trans

    def voting(self, models, n_classes, X, weights=None):
        if weights is None:
            weights = [1 for i in range(len(models))]

        for i in range(len(models)):
            models[i].module.eval()

        preds = []
        for i_model, model in enumerate(models):
            preds.append(torch.argmax(nn.functional.softmax(model.module(X.cuda()), dim=1), dim=1))

        pred_out = torch.zeros([X.shape[0], n_classes])
        for j, pred in enumerate(preds):
            for i, _ in enumerate(pred_out):
                pred_out[i][pred[i]] += weights[j]

        pred_out = torch.argmax(pred_out, dim=1)

        for i in range(len(models)):
            models[i].module.train()

        return pred_out


class SkorchRandSearch(SkorchFactory):
    def __init__(self,Model,**kwargs):
        super(SkorchRandSearch, self).__init__(Model=Model,**kwargs)
        if 'n_iter' in kwargs.keys():
            self.n_iter = kwargs['n_iter']
        else:
            self.n_iter = 4

    def compile(self, model):
        Trainner = NeuralNetClassifier(
            model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            device=self.cuda,
            max_epochs=self.max_epochs,
            callbacks=[
                # self.tensorboard_cb,
                self.early_stopping,
                self.gradient_norm_clipping,
                ('lr_scheduler', self.cosanllr)
            ],
            batch_size=16,
        )
        return Trainner

    def search(self, params, model, verbose=False):
        Trainner = self.compile(model)
        Trainner.verbose = verbose

        gs = RandomizedSearchCV(Trainner, params, n_iter=self.n_iter, refit=False, cv=3, scoring='accuracy', verbose=4)
        return gs



class SkorchGridSearch(SkorchFactory):
    def __init__(self,Model,**kwargs):
        super(SkorchGridSearch, self).__init__(Model=Model,**kwargs)

    def compile(self, model):
        Trainner = NeuralNetClassifier(
            model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            device=self.cuda,
            max_epochs=self.max_epochs,
            callbacks=[
                # self.tensorboard_cb,
                self.early_stopping,
                self.gradient_norm_clipping,
                ('lr_scheduler', self.cosanllr)
            ],
            batch_size=16,
        )
        return Trainner

    def search(self,params,model,verbose=False):
        Trainner = self.compile(model)
        Trainner.verbose = verbose

        gs = GridSearchCV(Trainner, params, refit=False, cv=3, scoring='accuracy')
        return gs

    def return_df_search(self, searcher, keys):
        data = searcher.cv_results_
        metrics = []
        for key in keys:
            metrics.append('param_' + key)
        metrics.append('mean_test_score')
        df = pd.DataFrame({key: value for (key, value) in data.items() if key in metrics})
        return df


