import pickle
from copy import deepcopy
import numpy as np
import torch
from sklearn.model_selection import KFold
import random
from abc import ABCMeta,abstractmethod
from datetime import datetime
import os
import pandas as pd

from PluginCore.Processor.base import preprocess,_preprocess
from PluginCore.Datasets.utils.windowers import create_windows_from_events
from PluginCore.Datasets.utils.Xy import X_y_from_Dataset,X_from_Dataset,X_y_ID_from_Dataset
from PluginCore.Datasets.base import BaseConcatDataset,SubjectIDDataset
from PluginCore.Datasets.moabb import get_dataset_rest_stated
from PluginCore.Decoder.trainer.utils import return_df_search



class PluginCore():
    def __init__(self,preprocess,algorithms,datasets,inspectors=None,modules={}):
        self.preprocess = preprocess
        self.algorithms = algorithms
        self.datasets = datasets
        self.inspectors = inspectors
        self.modules = modules

    def check_train_mode(self, dataset, subject_mode, train_mode, train_subjects, valid_subjects, test_subject):
        """
        Check if arguments are right.

        :param dataset:
        :param subject_mode:
        :param train_mode:
        :param train_subjects:
        :param valid_subjects:
        :param test_subject:
        :return:
        """
        assert subject_mode in ['subject_dependent', 'subject_independent_random', 'subject_transfer_unlabel',
                                'subject_transfer_label', 'subject_transfer']
        assert train_mode in ['hold_out', 'cross_validation', 'nested_cross_validation']

        if subject_mode == 'subject_dependent':
            # Check data from same subject
            assert len(np.unique([s for s in dataset.description['subject']])) == 1, print(
                'Can not pass multiple subjects dataset if using subject-independent')

        if subject_mode in ['subject_transfer_unlabel', 'subject_transfer']:
            if train_mode=='hold_out':
                assert train_subjects is not None
                assert test_subject is not None

        if subject_mode is 'subject_transfer_label':
            if train_mode=='hold_out':
                assert test_subject is not None

    def provide_Xys(self,dataset_id,preprocess_id,subject_mode,train_mode,trial_start_offset_seconds,
                    trial_end_offset_seconds,train_r,n_fold,
                    train_subjects=None,valid_subjects=None,test_subject=None,direct_window=False):
        """
        From inner dataset provide X-y pair for machine learning models, based on training mode
        :param dataset_id:
        :param preprocess_id:
        :param subject_mode:
        :param train_mode:
        :param trial_start_offset_seconds:
        :param trial_end_offset_seconds:
        :param train_r:
        :param n_fold:
        :param train_subjects:
        :param valid_subjects:
        :param test_subject:
        :param direct_window:
        :return:
        """
        dataset = deepcopy(self.datasets[dataset_id])
        if direct_window:
            _preprocess(dataset.windows, self.preprocess[preprocess_id])
        else:
            preprocess(dataset, self.preprocess[preprocess_id])
        return self._provide_Xys(dataset=dataset,subject_mode=subject_mode,train_mode=train_mode,trial_start_offset_seconds=trial_start_offset_seconds,
                    trial_end_offset_seconds=trial_end_offset_seconds,train_r=train_r,n_fold=n_fold,
                    train_subjects=train_subjects,valid_subjects=valid_subjects,test_subject=test_subject,direct_window=direct_window)

    def _provide_Xys(self,dataset,subject_mode,train_mode,trial_start_offset_seconds,
                    trial_end_offset_seconds,train_r,n_fold,
                    train_subjects=None,valid_subjects=None,test_subject=None,
                     direct_window=False):

        if direct_window:
            sfreq = dataset.windows[0].info['sfreq']
        else:
            sfreq = dataset.datasets[0].raw.info['sfreq']
            assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_end_offset_samples = int(trial_end_offset_seconds * sfreq)

        if subject_mode in ['subject_dependent', 'subject_independent_random']:
            # Create window dataset
            if direct_window:
                windows_dataset = deepcopy(dataset)
            else:
                windows_dataset = create_windows_from_events(
                    dataset,
                    trial_start_offset_samples=trial_start_offset_samples,
                    trial_stop_offset_samples=trial_end_offset_samples,
                    preload=True,
                )

            if train_mode is 'hold_out':
                # Create train_set and valid_set, here valid_set is used for test
                n_trials = len(windows_dataset)
                n_trials_train = int(train_r * n_trials)
                n_trials_test = n_trials - n_trials_train
                train_set, test_set = torch.utils.data.random_split(windows_dataset, [n_trials_train, n_trials_test])

                train_X, train_y = X_y_from_Dataset(train_set)
                test_X, test_y = X_y_from_Dataset(test_set)

                return (train_X,train_y),(test_X,test_y)

            elif train_mode is 'cross_validation':
                all_X, all_y = X_y_from_Dataset(windows_dataset)
                kfold = KFold(n_splits=n_fold, shuffle=False)

                train_Xs,train_ys,test_Xs,test_ys = [],[],[],[]

                for i_cv, (train_idx, test_idx) in enumerate(kfold.split(all_X, all_y)):
                    train_X, train_y = all_X[train_idx], all_y[train_idx]
                    valid_X, valid_y = all_X[test_idx], all_y[test_idx]
                    train_Xs.append(train_X)
                    train_ys.append(train_y)
                    test_Xs.append(valid_X)
                    test_ys.append(valid_y)
                return (train_Xs,train_ys),(test_Xs,test_ys)

            elif train_mode is 'nested_cross_validation':
                all_X, all_y = X_y_from_Dataset(windows_dataset)
                kfold = KFold(n_splits=n_fold, shuffle=False)

                train_Xs, train_ys, valid_Xs, valid_ys, test_Xs, test_ys = [], [], [], [], [], []

                for i_cv, (train_idx, test_idx) in enumerate(kfold.split(all_X, all_y)):
                    train_X, train_y = all_X[train_idx], all_y[train_idx]
                    test_X, test_y = all_X[test_idx], all_y[test_idx]

                    train_Xs_inner, train_ys_inner, valid_Xs_inner, valid_ys_inner, test_Xs_inner, test_ys_inner = [], [], [], [], [], []
                    kfold_inner = KFold(n_splits=n_fold, shuffle=False)
                    for i_cv_inner, (train_idx_inner, test_idx_inner) in enumerate(kfold_inner.split(train_X, train_y)):
                        train_X_inner, train_y_inner = train_X[train_idx_inner], train_y[train_idx_inner]
                        valid_X_inner, valid_y_inner = train_X[test_idx_inner], train_y[test_idx_inner]
                        train_Xs_inner.append(train_X_inner)
                        train_ys_inner.append(train_y_inner)
                        valid_Xs_inner.append(valid_X_inner)
                        valid_ys_inner.append(valid_y_inner)
                        test_Xs_inner.append(test_X)
                        test_ys_inner.append(test_y)

                    train_Xs.append(train_Xs_inner)
                    train_ys.append(train_ys_inner)
                    valid_Xs.append(valid_Xs_inner)
                    valid_ys.append(valid_ys_inner)
                    test_Xs.append(test_Xs_inner)
                    test_ys.append(test_ys_inner)

                return (train_Xs,train_ys),(valid_Xs,valid_ys),(test_Xs,test_ys)

        if subject_mode is 'subject_transfer_label':
            assert (direct_window is False), print("direct window on this mode not implemented")
            if train_mode is 'hold_out':
                dataset_split = dataset.split('subject')
                target_set = create_windows_from_events(
                    dataset_split[str(test_subject)],
                    trial_start_offset_samples=trial_start_offset_samples,
                    trial_stop_offset_samples=trial_end_offset_samples,
                    preload=True,
                )
                n_trials = len(target_set)
                n_trials_train = int(train_r * n_trials)
                n_trials_test = n_trials - n_trials_train
                train_set, test_set = torch.utils.data.random_split(target_set, [n_trials_train, n_trials_test])

                train_X, train_y = X_y_from_Dataset(train_set)
                test_X, test_y = X_y_from_Dataset(test_set)
                return (train_X,train_y), (test_X, test_y)

        if subject_mode in ['subject_transfer']:
            assert (direct_window is False), print("direct window on this mode not implemented")
            if train_mode is 'hold_out':
                dataset_split = dataset.split('subject')
                test_set = create_windows_from_events(
                    dataset_split[str(test_subject)],
                    trial_start_offset_samples=trial_start_offset_samples,
                    trial_stop_offset_samples=trial_end_offset_samples,
                    preload=True,
                )
                test_X, test_y = X_y_from_Dataset(test_set)

                train_set = create_windows_from_events(
                    BaseConcatDataset([dataset_split[d] for d in dataset_split if int(d) in train_subjects]),
                    trial_start_offset_samples=trial_start_offset_samples,
                    trial_stop_offset_samples=trial_end_offset_samples,
                    preload=True,
                )
                train_X, train_y = X_y_from_Dataset(train_set)

                return (train_X,train_y),(test_X,test_y)

            elif train_mode is 'cross_validation':
                train_Xs,train_ys,test_Xs,test_ys = [],[],[],[]

                subjects = np.unique([str(s) for s in dataset.description['subject']])
                subjects = list(subjects)

                for sub_i in subjects:
                    test_subject = sub_i
                    train_subjects = deepcopy(subjects)
                    train_subjects.remove(sub_i)

                    dataset_split = dataset.split('subject')
                    test_set = create_windows_from_events(
                        dataset_split[str(test_subject)],
                        trial_start_offset_samples=trial_start_offset_samples,
                        trial_stop_offset_samples=trial_end_offset_samples,
                        preload=True,
                    )
                    test_X, test_y = X_y_from_Dataset(test_set)

                    train_set = create_windows_from_events(
                        BaseConcatDataset([dataset_split[d] for d in dataset_split if d in train_subjects]),
                        trial_start_offset_samples=trial_start_offset_samples,
                        trial_stop_offset_samples=trial_end_offset_samples,
                        preload=True,
                    )
                    train_X, train_y = X_y_from_Dataset(train_set)

                    train_Xs.append(train_X)
                    train_ys.append(train_y)
                    test_Xs.append(test_X)
                    test_ys.append(test_y)

                return (train_Xs,train_ys),(test_Xs,test_ys),subjects

    def _provide_Xs(self, dataset, window_seconds):

        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        window_seconds_sample = int(sfreq * window_seconds)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=window_seconds_sample,
            window_stride_samples=10,
            preload=True,
        )

        rest_X = X_from_Dataset(windows_dataset,shuffle=False)

        return rest_X

    def provide_Xs(self, dataset_id, preprocess_id, subject_ids, window_seconds):
        dataset = deepcopy(self.datasets[dataset_id])
        preprocess(dataset,self.preprocess[preprocess_id])
        rest_state_dataset = get_dataset_rest_stated(dataset)


        dataset_split = rest_state_dataset.split('subject')
        target_dataset = BaseConcatDataset([dataset_split[d] for d in dataset_split if int(d) in subject_ids])

        return self._provide_Xs(dataset=target_dataset,window_seconds=window_seconds)

    def set_random_seed_for_train(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def search_model(self, preprocesser_id, algorithm_id, dataset_id, model, params, subject_mode,
                     trial_start_offset_seconds,trial_end_offset_seconds, seed=2022, verbose=False, direct_window=False):
        assert subject_mode in ['subject_dependent','subject_independent_random']
        dataset = deepcopy(self.datasets[dataset_id])
        if direct_window:
            _preprocess(dataset.windows, self.preprocess[preprocesser_id])
        else:
            preprocess(dataset, self.preprocess[preprocesser_id])
        self.set_random_seed_for_train(seed)

        (all_X, all_y), _ = self._provide_Xys(dataset=dataset, subject_mode=subject_mode,
                                                                train_mode='hold_out', direct_window=direct_window,
                                                                trial_start_offset_seconds=trial_start_offset_seconds,
                                                                trial_end_offset_seconds=trial_end_offset_seconds,
                                                                train_r=0.99, n_fold=None)

        searcher = self.algorithms[algorithm_id]
        gs = searcher.search(params=params, model=model, verbose=verbose)
        gs.fit(all_X, all_y)

        return gs

    def train_model(self, preprocesser_id, algorithm_id, dataset_id, model,
                    subject_mode, train_mode, trial_start_offset_seconds,trial_end_offset_seconds,train_r,n_fold,seed=2022,verbose=True,
                    train_subjects=None,valid_subjects=None,test_subject=None,score_on_train=False, direct_window=False,
                    rest_state_window_seconds=None):
        dataset = deepcopy(self.datasets[dataset_id])
        if direct_window:
            _preprocess(dataset.windows, self.preprocess[preprocesser_id])
        else:
            preprocess(dataset, self.preprocess[preprocesser_id])

        self.check_train_mode(dataset=dataset, subject_mode=subject_mode, train_mode=train_mode,
                              train_subjects=train_subjects, valid_subjects=valid_subjects, test_subject=test_subject)
        self.set_random_seed_for_train(seed)

        if subject_mode in ['subject_dependent','subject_independent_random']:

            if train_mode is 'hold_out':
                (train_X,train_y),(test_X,test_y) = self._provide_Xys(dataset=dataset,subject_mode=subject_mode,train_mode=train_mode,
                    trial_start_offset_seconds=trial_start_offset_seconds,trial_end_offset_seconds=trial_end_offset_seconds,train_r=train_r,n_fold=n_fold,
                    train_subjects=train_subjects,valid_subjects=valid_subjects,test_subject=test_subject,direct_window=direct_window)

                model_tmp = deepcopy(model)

                algorithm = self.algorithms[algorithm_id]
                model_tmp = algorithm.train(train_X, train_y, verbose=verbose, model=model_tmp, log_dir='runs')

                if score_on_train:
                    #TODO:change it to algorithm.score
                    score = algorithm.score(model=model_tmp, test_X=test_X, test_y=test_y)
                    print('valid acc using hold-out:', score)
                return model_tmp, (train_X, train_y), (test_X, test_y)

            elif train_mode is 'cross_validation':
                (train_Xs, train_ys), (test_Xs, test_ys) = self._provide_Xys(dataset=dataset,subject_mode=subject_mode,train_mode=train_mode,
                    trial_start_offset_seconds=trial_start_offset_seconds,trial_end_offset_seconds=trial_end_offset_seconds,train_r=train_r,n_fold=n_fold,
                    train_subjects=train_subjects,valid_subjects=valid_subjects,test_subject=test_subject,direct_window=direct_window)

                trained_models = []
                assert len(train_Xs)==n_fold and len(train_ys)==n_fold and len(test_Xs)==n_fold and len(test_ys)==n_fold
                for i_fold in range(n_fold):
                    train_X, train_y, test_X, test_y = train_Xs[i_fold], train_ys[i_fold], test_Xs[i_fold], test_ys[i_fold]

                    algorithm = self.algorithms[algorithm_id]

                    model_tmp = deepcopy(model)
                    model_tmp = algorithm.train(train_X, train_y, verbose=verbose, model=model_tmp, log_dir='runs')

                    if score_on_train:
                        test_score = algorithm.score(model=model_tmp, test_X=test_X, test_y=test_y)
                        print('valid acc using hold-out on fold ', i_fold+1 , ': ', test_score)

                    trained_models.append(model_tmp)

                return trained_models, (train_Xs,train_ys), (test_Xs,test_ys)

            elif train_mode is 'nested_cross_validation':
                (train_Xs, train_ys), (valid_Xs, valid_ys), (test_Xs, test_ys) = self._provide_Xys(dataset=dataset,subject_mode=subject_mode,train_mode=train_mode,
                    trial_start_offset_seconds=trial_start_offset_seconds,trial_end_offset_seconds=trial_end_offset_seconds,train_r=train_r,n_fold=n_fold,
                    train_subjects=train_subjects,valid_subjects=valid_subjects,test_subject=test_subject,direct_window=direct_window)


                test_accs = []
                trained_models = []

                trainned_Xs = []; trainned_ys = []; valided_Xs = []; valided_ys = []; tested_Xs = []; tested_ys = []
                for i_fold in range(n_fold):
                    train_X, train_y, valid_X, valid_y, test_X, test_y = \
                        train_Xs[i_fold], train_ys[i_fold], valid_Xs[i_fold], valid_ys[i_fold], test_Xs[i_fold], test_ys[i_fold]

                    test_accs_inner = []
                    trained_models_inner = []
                    for i_fold_inner in range(n_fold):
                        train_X_inner, train_y_inner, valid_X_inner, valid_y_inner, test_X_inner, test_y_inner = \
                            train_X[i_fold], train_y[i_fold], valid_X[i_fold], valid_y[i_fold], test_X[i_fold], test_y[i_fold]

                        algorithm = self.algorithms[algorithm_id]

                        model_tmp = deepcopy(model)
                        model_tmp = algorithm.train(train_X_inner, train_y_inner, verbose=verbose, model=model_tmp, log_dir='runs')

                        if score_on_train:
                            test_score = algorithm.score(test_X_inner, test_y_inner)
                            test_accs_inner.append(test_score)
                        trained_models_inner.append(model_tmp)
                        trainned_Xs.append(train_X_inner);trainned_ys.append(train_y_inner);valided_Xs.append(valid_X_inner);valided_ys.append(valid_y_inner)
                        tested_Xs.append(test_X_inner);tested_ys.append(test_y_inner)
                    if score_on_train:
                        test_accs.append(test_accs_inner)
                    trained_models.append(trained_models_inner)

                return trained_models, (trainned_Xs, trainned_ys), (valided_Xs,valided_ys), (tested_Xs,tested_ys)

        if subject_mode in ['subject_transfer']:
            # train in non-target data mode
            if train_mode is 'hold_out':
                (train_X, train_y), (test_X, test_y) = self._provide_Xys(dataset=dataset, subject_mode=subject_mode,
                                                                        train_mode=train_mode,
                                                                        trial_start_offset_seconds=trial_start_offset_seconds,
                                                                        trial_end_offset_seconds=trial_end_offset_seconds,
                                                                        train_r=train_r, n_fold=n_fold,
                                                                        train_subjects=train_subjects,
                                                                        valid_subjects=valid_subjects,
                                                                        test_subject=test_subject)

                model_tmp = deepcopy(model)

                algorithm = self.algorithms[algorithm_id]
                model_tmp = algorithm.train(train_X, train_y, verbose=verbose, model=model_tmp, log_dir='runs')

                if score_on_train:
                    print('valid acc using hold-out:', algorithm.score(model=model_tmp, test_X=test_X, test_y=test_y))
                return model_tmp, (train_X, train_y), (test_X, test_y)

            elif train_mode is 'cross_validation':
                (train_Xs, train_ys), (test_Xs, test_ys), subjects = self._provide_Xys(dataset=dataset, subject_mode=subject_mode,
                                                                            train_mode=train_mode,
                                                                            trial_start_offset_seconds=trial_start_offset_seconds,
                                                                            trial_end_offset_seconds=trial_end_offset_seconds,
                                                                            train_r=train_r, n_fold=n_fold,
                                                                            train_subjects=train_subjects,
                                                                            valid_subjects=valid_subjects,
                                                                            test_subject=test_subject)

                test_accs = []
                trained_models = []

                for i_fold,sub in enumerate(subjects):
                    train_X, train_y, test_X, test_y = train_Xs[i_fold], train_ys[i_fold], test_Xs[i_fold], test_ys[
                        i_fold]

                    algorithm = self.algorithms[algorithm_id]

                    model_tmp = deepcopy(model)
                    model_tmp = algorithm.train(train_X, train_y, verbose=verbose, model=model_tmp, log_dir='runs')

                    if score_on_train:
                        test_score = algorithm.score(model=model_tmp, test_X=test_X, test_y=test_y)
                        print('valid acc using hold-out on fold ', i_fold + 1, ': ', test_score)
                        test_accs.append(test_score)
                    trained_models.append(model_tmp)

                return trained_models, subjects, (test_Xs,test_ys)

        if subject_mode in ['subject_transfer_unlabel']:
            assert rest_state_window_seconds is not None, print('Should provide window length for rest-state data')
            if train_mode is 'hold_out':
                # in here train_r refers to train_ratio of target dataset
                (train_X, train_y), (test_X, test_y) = self._provide_Xys(dataset=dataset,subject_mode='subject_transfer',train_mode='hold_out',
                                                         trial_start_offset_seconds=trial_start_offset_seconds,
                                                         trial_end_offset_seconds=trial_end_offset_seconds,
                                                         train_r=train_r, n_fold=n_fold,
                                                         train_subjects=train_subjects,
                                                         valid_subjects=valid_subjects,
                                                         test_subject=test_subject)
                rest_X = self.provide_Xs(dataset_id=dataset_id,preprocess_id=preprocesser_id,subject_ids=[test_subject],window_seconds=rest_state_window_seconds)

                model_tmp = deepcopy(model)

                algorithm = self.algorithms[algorithm_id]
                model_ori, model_trans = algorithm.train_adapt(train_X=train_X, train_y=train_y, verbose=verbose, model=model_tmp,
                                                  log_dir='runs', rest_X=rest_X)

                if score_on_train:
                    score_ori = algorithm.score(model=model_ori, test_X=test_X, test_y=test_y)
                    score_trans = algorithm.score(model=model_trans, test_X=test_X, test_y=test_y)
                    print('valid acc using hold-out, before adapt:', score_ori)
                    print('valid acc using hold-out, after adapt:', score_trans)
                return (model_ori, model_trans), (train_X, train_y), (test_X, test_y)

            elif train_mode is 'cross_validation':
                (train_Xs, train_ys), (test_Xs, test_ys), subjects = self._provide_Xys(dataset=dataset, subject_mode='subject_transfer',
                                                                            train_mode=train_mode,
                                                                            trial_start_offset_seconds=trial_start_offset_seconds,
                                                                            trial_end_offset_seconds=trial_end_offset_seconds,
                                                                            train_r=train_r, n_fold=n_fold,
                                                                            train_subjects=train_subjects,
                                                                            valid_subjects=valid_subjects,
                                                                            test_subject=test_subject)

                rest_data = [self.provide_Xs(dataset_id=dataset_id,preprocess_id=preprocesser_id,subject_ids=[int(s)],window_seconds=rest_state_window_seconds)
                                 for s in subjects]

                test_accs = []
                trained_models = []

                for i_fold,sub in enumerate(subjects):
                    train_X, train_y = train_Xs[i_fold], train_ys[i_fold]
                    test_X, test_y = test_Xs[i_fold], test_ys[i_fold]
                    rest_X = rest_data[i_fold]

                    algorithm = self.algorithms[algorithm_id]

                    model_tmp = deepcopy(model)
                    model_tmp = algorithm.train_adapt(train_X=train_X, train_y=train_y, rest_X=rest_X,
                                                      verbose=verbose, model=model_tmp, log_dir='runs')

                    if score_on_train:
                        test_score = algorithm.score(model=model_tmp, test_X=test_X, test_y=test_y)
                        print('valid acc using hold-out on fold ', i_fold + 1, ': ', test_score)
                        test_accs.append(test_score)
                    trained_models.append(model_tmp)

                return trained_models, subjects, (test_Xs,test_ys)

        if subject_mode in ['subject_transfer_label']:
            # train in little target data mode
            if train_mode is 'hold_out':
                # in here train_r refers to train_ratio of target dataset
                (train_X, train_y), _ = self._provide_Xys(dataset=dataset,
                                                                         subject_mode='subject_transfer',
                                                                         train_mode='hold_out',
                                                                         trial_start_offset_seconds=trial_start_offset_seconds,
                                                                         trial_end_offset_seconds=trial_end_offset_seconds,
                                                                         train_r=train_r, n_fold=n_fold,
                                                                         train_subjects=train_subjects,
                                                                         valid_subjects=valid_subjects,
                                                                         test_subject=test_subject)
                (little_X,little_y), (test_X, test_y) = self.provide_Xys(dataset_id=dataset_id,preprocess_id=preprocesser_id,subject_mode='subject_transfer_label',
                                                                         train_mode='hold_out',trial_start_offset_seconds=trial_start_offset_seconds,
                                                                         trial_end_offset_seconds=trial_end_offset_seconds,train_r=train_r,n_fold=None,
                                                                        test_subject=test_subject)

                model_tmp = deepcopy(model)

                algorithm = self.algorithms[algorithm_id]
                model_ori, model_trans = algorithm.train_finetune(train_X=train_X, train_y=train_y, verbose=verbose, model=model_tmp,
                                                  log_dir='runs', little_X=little_X, little_y=little_y)

                if score_on_train:
                    score_ori = algorithm.score(model=model_ori, test_X=test_X, test_y=test_y)
                    score_trans = algorithm.score(model=model_trans, test_X=test_X, test_y=test_y)
                    print('valid acc using hold-out, before finetune:', score_ori)
                    print('valid acc using hold-out, after finetune:', score_trans)
                return (model_ori, model_trans), (train_X, train_y), (test_X, test_y)

            elif train_mode is 'cross_validation':
                (train_Xs, train_ys), (test_Xs, test_ys), subjects = self._provide_Xys(dataset=dataset, subject_mode='subject_transfer',
                                                                            train_mode=train_mode,
                                                                            trial_start_offset_seconds=trial_start_offset_seconds,
                                                                            trial_end_offset_seconds=trial_end_offset_seconds,
                                                                            train_r=train_r, n_fold=n_fold,
                                                                            train_subjects=train_subjects,
                                                                            valid_subjects=valid_subjects,
                                                                            test_subject=test_subject)

                finetune_data = [self.provide_Xys(dataset_id=dataset_id,preprocess_id=preprocesser_id,subject_mode='subject_transfer_label',
                                        train_mode='hold_out',trial_start_offset_seconds=trial_start_offset_seconds,
                                        trial_end_offset_seconds=trial_end_offset_seconds,train_r=train_r, n_fold=None,test_subject=s)
                                 for s in subjects]

                test_accs = []
                trained_models = []

                for i_fold,sub in enumerate(subjects):
                    train_X, train_y = train_Xs[i_fold], train_ys[i_fold]
                    (little_X,little_y), (test_X, test_y) = finetune_data[i_fold]

                    algorithm = self.algorithms[algorithm_id]

                    model_tmp = deepcopy(model)
                    model_tmp = algorithm.train_finetune(train_X=train_X, train_y=train_y, little_X=little_X,
                                                         little_y=little_y, verbose=verbose, model=model_tmp, log_dir='runs')

                    if score_on_train:
                        test_score = algorithm.score(model=model_tmp, test_X=test_X, test_y=test_y)
                        print('valid acc using hold-out on fold ', i_fold + 1, ': ', test_score)
                        test_accs.append(test_score)
                    trained_models.append(model_tmp)

                return trained_models, subjects, (test_Xs,test_ys)

    def inspect(self, ans, subject_mode, train_mode, inspector_id):
        inspector = self.inspectors[inspector_id]
        if subject_mode in ['subject_dependent', 'subject_independent_random']:
            if train_mode is 'hold_out':
                model, (test_X, test_y) = ans[0] , ans[2]
                re = inspector.inspect(test_X, test_y, model)
                return re

            elif train_mode is 'cross_validation':
                trained_models, (test_Xs, test_ys) = ans[0], ans[2]
                n_fold = len(ans[0])
                res = []
                for i_fold in range(n_fold):
                    re = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], trained_models[i_fold])
                    res.append(re)
                return res

            elif train_mode is 'nested_cross_validation':
                trained_models, (test_Xs, test_ys), (valid_Xs, valid_ys) = ans[0], ans[3], ans[2]
                n_fold = len(ans[0])

                test_res = []
                for i_fold in range(n_fold):
                    re = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], trained_models[i_fold])
                    test_res.append(re)

                valid_res = []
                for i_fold in range(n_fold):
                    re = inspector.inspect(valid_Xs[i_fold], valid_ys[i_fold], trained_models[i_fold])
                    valid_res.append(re)

                return (valid_res, test_res)

        if subject_mode in ['subject_transfer']:
            if train_mode is 'hold_out':
                model, (test_X, test_y) = ans[0], ans[2]
                re = inspector.inspect(test_X, test_y, model)
                return re

            elif train_mode is 'cross_validation':
                trained_models, subjects, (test_Xs, test_ys) = ans[0], ans[1], ans[2]
                res = []
                for i_fold, sub in enumerate(subjects):
                    re = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], trained_models[i_fold])
                    res.append(re)

                return res

        if subject_mode in ['subject_transfer_unlabel']:
            if train_mode is 'hold_out':
                models, (test_X, test_y) = ans[0], ans[2]
                model_ori = models[0]
                model_trans = models[1]
                re_ori = inspector.inspect(test_X, test_y, model_ori)
                re_trans = inspector.inspect(test_X, test_y, model_trans)
                return (re_ori, re_trans)

            elif train_mode is 'cross_validation':
                trained_models, subjects, (test_Xs, test_ys) = ans[0], ans[1], ans[2]
                res = []
                for i_fold, sub in enumerate(subjects):
                    model_ori = trained_models[i_fold][0]
                    model_trans = trained_models[i_fold][1]
                    re_ori = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], model_ori)
                    re_trans = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], model_trans)
                    res.append((re_ori, re_trans))

                return res

        if subject_mode in ['subject_transfer_label']:
            if train_mode is 'hold_out':
                models, (test_X, test_y) = ans[0], ans[2]
                model_ori = models[0]
                model_trans = models[1]
                re_ori = inspector.inspect(test_X, test_y, model_ori)
                re_trans = inspector.inspect(test_X, test_y, model_trans)
                return (re_ori, re_trans)

            elif train_mode is 'cross_validation':
                trained_models, subjects, (test_Xs, test_ys) = ans[0], ans[1], ans[2]
                res = []
                for i_fold, sub in enumerate(subjects):
                    model_ori = trained_models[i_fold][0]
                    model_trans = trained_models[i_fold][1]
                    re_ori = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], model_ori)
                    re_trans = inspector.inspect(test_Xs[i_fold], test_ys[i_fold], model_trans)
                    res.append((re_ori,re_trans))

                return res

    def track_time(self, preprocesser_id, algorithm_id, dataset_id, session_id,
                   trial_start_offset_seconds, trial_end_offset_seconds, model, n_inter,direct_window=False):
        dataset = deepcopy(self.datasets[dataset_id])
        if direct_window:
            _preprocess(dataset.windows, self.preprocess[preprocesser_id])

            windows_dataset = dataset
        else:
            preprocess(dataset, self.preprocess[preprocesser_id])

            dataset_session = dataset.split('session')[session_id]
            trial_start_offset_seconds = trial_start_offset_seconds
            trial_end_offset_seconds = trial_end_offset_seconds

            sfreq = dataset.datasets[0].raw.info['sfreq']
            assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
            trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
            trial_end_offset_samples = int(trial_end_offset_seconds * sfreq)

            windows_dataset = create_windows_from_events(
                dataset_session,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=trial_end_offset_samples,
                preload=True,
            )

        all_X, all_y = X_y_from_Dataset(windows_dataset)
        self.algorithms[algorithm_id].compile(model)
        self.algorithms[algorithm_id].track_time(all_X=all_X,all_y=all_y,n_inter=n_inter)

    def feature_analysis_wrapper(self, preprocess_id, algorithm_id ,dataset_id, subject_mode, train_mode, trial_start_offset_seconds,
                                 trial_end_offset_seconds, train_r, n_fold, model, sub_channels, time_intervals, viz_metric=None, direct_window=False):
        ans = self.provide_Xys(preprocess_id=preprocess_id, dataset_id=dataset_id,subject_mode=subject_mode, train_mode=train_mode,
                                    trial_start_offset_seconds=trial_start_offset_seconds, trial_end_offset_seconds=trial_end_offset_seconds,
                                    train_r=train_r, n_fold=n_fold, direct_window=direct_window)
        if subject_mode=='subject_dependent' and train_mode=='hold_out':
            res = []
            for sub_channel in sub_channels:
                for time_interval in time_intervals:
                    (train_X, train_y), (test_X, test_y) = ans[0], ans[1]
                    sub_model = self.algorithms[algorithm_id].compile(model, sub_channel, time_interval)

                    trainned_model = self.algorithms[algorithm_id].train(model=sub_model, train_X=train_X, train_y=train_y, channels=sub_channel,
                                                       time_interval=time_interval)
                    re = self.algorithms[algorithm_id].inspect(test_X=test_X, test_y=test_y, model=trainned_model, channels=sub_channel,
                                             time_interval=time_interval)
                    res.append((re, sub_channel, time_interval))
            if viz_metric is not None:
                self.algorithms[algorithm_id].plot_metric(res=res,metric=viz_metric)
            return res

    def log_model_search(self, gs,  keys, dir):
        df = return_df_search(gs, keys)

        time_info = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_name = 'MS_' + time_info + '.csv'  # MRS: model search
        csv_path = dir + log_name
        if os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, mode='a', header=False)
        else:
            df.to_csv(csv_path, index=False, mode='w', header=True)

    def log_subject_dependent(self, df, dataset_name, dir):
        self._log_df_to_csv(prefix='SDC', df=df, dataset_name=dataset_name, dir=dir)

    def log_none_target_data(self, df, dataset_name, dir):
        self._log_df_to_csv(prefix='NTC', df=df, dataset_name=dataset_name, dir=dir)

    def log_rest_target_data(self, df, dataset_name, dir):
        self._log_df_to_csv(prefix='RTC', df=df, dataset_name=dataset_name, dir=dir)

    def log_little_target_data(self, df, dataset_name, dir, trainned_models=None, data=None):
        self._log_df_to_csv(prefix='LTC', df=df, dataset_name=dataset_name, dir=dir)

    def _log_df_to_csv(self, prefix, df, dataset_name, dir):
        time_info = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_name = prefix + time_info + '_' + dataset_name + '.csv'  # NTC: Non-Target data Compare
        csv_path = dir + log_name
        if os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, mode='a', header=False)
        else:
            df.to_csv(csv_path, index=False, mode='w', header=True)

    def run_cv_subject_dependent(self, preprocesser_id, algorithm_id, inspector_id, model, trial_start_offset_seconds,
                          trial_end_offset_seconds, n_fold, model_name, metrics, direct_window=False):
        self.n_subjects = len(self.datasets)
        self.subject_list = self.datasets.keys()

        re_subjects = []
        trainned_model_subjects = []
        for sub in self.subject_list:
            ans = self.train_model(preprocesser_id=preprocesser_id,algorithm_id=algorithm_id,dataset_id=sub,model=model,
                                   subject_mode='subject_dependent',train_mode='cross_validation',
                                   trial_start_offset_seconds=trial_start_offset_seconds,
                                   trial_end_offset_seconds=trial_end_offset_seconds,n_fold=n_fold,train_r=None,direct_window=direct_window)
            res = self.inspect(ans, subject_mode='subject_dependent',train_mode='cross_validation',inspector_id=inspector_id)
            for i_fold,re in enumerate(res):
                re['model'] = model_name
                re['subject'] = sub
                re['i_fold'] = i_fold
                re_subjects.append(re)
                trainned_model_subjects.append(ans[0][i_fold])

        df_subjects = pd.DataFrame()
        for re in re_subjects:
            df_ = pd.DataFrame({key: [value] for key, value in re.items() if key in metrics})
            df_subjects = df_subjects.append(df_)


        self.n_subjects = None
        self.subject_list = None
        return re_subjects, trainned_model_subjects, df_subjects

    def run_cv_none_target_data(self, preprocesser_id, algorithm_id, dataset_id , inspector_id, model, trial_start_offset_seconds,
                          trial_end_offset_seconds, n_fold, model_name, metrics):
        ans = self.train_model(preprocesser_id=preprocesser_id,algorithm_id=algorithm_id,dataset_id=dataset_id,model=model,
                               subject_mode='subject_transfer',train_mode='cross_validation',
                               trial_start_offset_seconds=trial_start_offset_seconds,
                               trial_end_offset_seconds=trial_end_offset_seconds,n_fold=n_fold,train_r=None,
                               train_subjects=None,valid_subjects=None,test_subject=None)
        res = self.inspect(ans, subject_mode='subject_transfer', train_mode='cross_validation',
                           inspector_id=inspector_id)

        re_subjects = []
        trainned_modes, subjects = ans[0], ans[1]
        for i_fold, re in enumerate(res):
            re['model'] = model_name
            re['subject'] = subjects[i_fold]
            re_subjects.append(re)

        df_subjects = pd.DataFrame()
        for re in re_subjects:
            df_ = pd.DataFrame({key: [value] for key, value in re.items() if key in metrics})
            df_subjects = df_subjects.append(df_)

        return re_subjects, trainned_modes, df_subjects

    def run_cv_none_target_data_id(self, preprocesser_id, algorithm_id, dataset_id , inspector_id, model, trial_start_offset_seconds,
                          trial_end_offset_seconds, model_name, metrics, subjects):
        re_subjects = []
        trainned_models = []
        for sub in subjects:
            test_sub = sub
            train_subs = [s for s in deepcopy(subjects) if s != sub]
            model_tmp = deepcopy(model)
            ans = self.train_model_with_id(preprocesser_id=preprocesser_id,algorithm_id=algorithm_id,dataset_id=dataset_id,model=model_tmp,
                                discriminator=None, trial_start_offset_seconds=trial_start_offset_seconds,
                                trial_end_offset_seconds=trial_end_offset_seconds, train_subjects=train_subs,test_subjects=test_sub)
            res = self.inspect(ans, subject_mode='subject_transfer', train_mode='hold_out',
                               inspector_id=inspector_id)

            trainned_models.append(ans[0])

            res['model'] = model_name
            res['subject'] = sub
            re_subjects.append(res)

        metrics.extend(['model','subject'])
        df_subjects = pd.DataFrame()
        for re in re_subjects:
            df_ = pd.DataFrame({key: [value] for key, value in re.items() if key in metrics})
            df_subjects = df_subjects.append(df_)

        return re_subjects, trainned_models, df_subjects

    def train_model_with_id(self, preprocesser_id, algorithm_id, dataset_id , model, discriminator ,trial_start_offset_seconds,
                          trial_end_offset_seconds, train_subjects, test_subjects, viz=False):
        dataset = deepcopy(self.datasets[dataset_id])
        preprocess(dataset,self.preprocess[preprocesser_id])

        sfreq = dataset.datasets[0].raw.info['sfreq']
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_end_offset_samples = int(trial_end_offset_seconds * sfreq)

        dataset_split = dataset.split('subject')
        train_set = create_windows_from_events(
            BaseConcatDataset([dataset_split[d] for d in dataset_split if int(d) in train_subjects]),
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_end_offset_samples,
            preload=True,
        )
        target_set = create_windows_from_events(
            dataset_split[str(test_subjects)],
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            preload=True,
        )
        subject_dataset = []
        for ds in train_set.datasets:
            subject_dataset.append(SubjectIDDataset(
                windows=ds.windows,
                description=ds.description,
                transform=ds.transform
            ))
        subject_dataset = BaseConcatDataset(subject_dataset)
        train_X, train_y, train_subjects = X_y_ID_from_Dataset(subject_dataset)
        test_X, test_y = X_y_from_Dataset(target_set)

        trainned_model = self.algorithms[algorithm_id].train_model_with_id(train_X=train_X, train_y=train_y, model=model, discriminator=discriminator,
                                                                train_subjects=train_subjects, test_X=test_X, test_y=test_y, viz=False)
        return trainned_model, (train_X, train_y), (test_X,test_y)

    def run_cv_rest_target_data(self, preprocesser_id, algorithm_id, dataset_id, model, trial_start_offset_seconds,
                          trial_end_offset_seconds, n_fold, inspector_id, model_name, metrics, rest_state_window_seconds=4):
        ans = self.train_model(preprocesser_id=preprocesser_id, algorithm_id=algorithm_id, dataset_id=dataset_id,
                               model=model, subject_mode='subject_transfer_unlabel', train_mode='cross_validation',
                               trial_start_offset_seconds=trial_start_offset_seconds, rest_state_window_seconds=rest_state_window_seconds,
                               trial_end_offset_seconds=trial_end_offset_seconds, n_fold=n_fold, train_r=None,
                               train_subjects=None, valid_subjects=None, test_subject=None)
        res = self.inspect(ans, subject_mode='subject_transfer_unlabel', train_mode='cross_validation',
                           inspector_id=inspector_id)

        re_subjects = []
        trainned_models, subjects = ans[0], ans[1]
        for i_fold, re_bi in enumerate(res):
            re_bi[0]['model'] = model_name
            re_bi[0]['subject'] = subjects[i_fold]
            re_bi[0]['state'] = 'BeforeAdapt'
            re_subjects.append(re_bi[0])

            re_bi[1]['model'] = model_name
            re_bi[1]['subject'] = subjects[i_fold]
            re_bi[1]['state'] = 'AfterAdapt'
            re_subjects.append(re_bi[1])

        df_subjects = pd.DataFrame()
        for re in re_subjects:
            df_ = pd.DataFrame({key: [value] for key, value in re.items() if key in metrics})
            df_subjects = df_subjects.append(df_)

        return re_subjects, trainned_models, df_subjects

    @staticmethod
    def print_csv_names(dir):
        files = os.listdir(dir)
        csv_files = [file for file in files if file.split('.')[-1] is '.csv' and file.split('_')[0]=='MS']
        print(csv_files)

    @staticmethod
    def read_df_from_file(file_path):
        df = pd.read_csv(file_path)
        return df

    def save_core(self, file_dir, file_name):
        document = {
            'datasets':self.datasets,
            'preprocess':self.preprocess,
            'algorithms':self.algorithms,
            'modules':self.modules,
            'inspectors':self.inspectors
        }
        with open(os.path.join(file_dir,file_name),'wb') as f:
            pickle.dump(document ,f)

    def load_core_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            document = pickle.load(f)
        self.datasets = document['datasets']
        self.preprocess = document['preprocess']
        self.algorithms = document['algorithms']
        self.modules = document['modules']
        self.inspectors = document['inspectors']

    def extract_raw_moabb(self, dataset_id, raw_id, preprocesser_id=None):
        dataset = deepcopy(self.datasets[dataset_id])
        if preprocesser_id is not None:
            preprocess(dataset, self.preprocess[preprocesser_id])
        raw = dataset.datasets[raw_id].raw
        return raw

    def extract_epoch_moabb(self, dataset_id, window_id, preprocesser_id=None):
        dataset = deepcopy(self.datasets[dataset_id])
        if preprocesser_id is not None:
            _preprocess(dataset, self.preprocess[preprocesser_id])
        windows = create_windows_from_events(concat_ds=dataset,
                                             trial_start_offset_samples=0,
                                             trial_stop_offset_samples=0,
                                             drop_last_window=True)
        epochs = windows.datasets[window_id].windows
        return epochs
