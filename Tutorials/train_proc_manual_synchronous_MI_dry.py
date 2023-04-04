import pickle
import os.path
import mne

from sklearn.svm import SVC
import seaborn as sns

from PluginCore.Processor.base import Processor
from PluginCore.Processor.module.temp import exponential_moving_standardize, convert_to_uV
from PluginCore.Decoder.trainer.Skorch import SkorchFactory, SkorchRandSearch, SkorchGridSearch
from PluginCore.Decoder.module.braindecode.eegnet import EEGNetv4
from PluginCore.Decoder.trainer.csp import CSPFactory, CSPRandomSearch, CSPFilter, CSPWrapper
from PluginCore.Decoder.inspector.csp import CSPInspector
from PluginCore.Decoder.inspector.Skorch import SkorchInspector
from PluginCore.Decoder.inspector.base import InspectorSyn
from PluginCore.Decoder.ml import classifibility
from PluginCore.base import PluginCore
from PluginCore.Decoder.module.braindecode.shallow_fbcsp import ShallowFBCSPNet
from PluginCore.Decoder.module.CSP.FBCSP import FBCSP, FBCSPSearch
from PluginCore.Decoder.trainer.utils import plot_res, boxplot_res, param3d_viz, return_df_search

file_dir = '.\\Tutorials\\data\\SynMI_Experiment\\'
dataset_name = 'MICalibrate_OpenBCI'

start_save_name = 'CoreModelStart.pkl'
compare_save_name = 'CoreModelCompare.pkl'
search_save_name = 'CoreModelSearch.pkl'
train_save_name = 'CoreModelTrain.pkl'
end_save_name = 'CoreModelEnd.pkl'


with open(os.path.join(file_dir,'Dateset_MICalibrate_OpenBCI.pkl'),'rb') as f:
    calibrate_data = pickle.load(f)


windowedDataset = calibrate_data['windowedDataset']
datasets = {
    1: windowedDataset,
}

# Define Preprocecss
low_cut_hz = 4.
high_cut_hz = 38.
factor_new = 1e-3
init_block_size = 1000

preps = {
    1: [Processor('pick', picks='eeg'),
        Processor(convert_to_uV),  # Convert from V to uV
        Processor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Processor(exponential_moving_standardize,  # Exponential moving standardization
                  factor_new=factor_new, init_block_size=init_block_size),
        ],
    2: [Processor('pick', picks='eeg')]
}

# Define algorithm
alg = {
    1: SkorchFactory(Model=ShallowFBCSPNet, max_epochs=600),
    2: CSPFactory(Model=FBCSP),
    3: SkorchRandSearch(Model=ShallowFBCSPNet, n_iter=10, max_epochs=250),
    4: SkorchGridSearch(Model=ShallowFBCSPNet),
    5: CSPRandomSearch(Model=FBCSP, n_iter=10),
    6: CSPFilter(filter_func=classifibility, model=None),
    7: CSPWrapper(train_factory=CSPFactory(Model=FBCSP), inspector_factory=CSPInspector(InspectorSyn(pos_label=None))),
}

# Define inspector
ispt = {
    1: SkorchInspector(InspectorSyn(pos_label=0, manual_n_classes=2), cuda=True),
    2: CSPInspector(InspectorSyn(pos_label=0, manual_n_classes=2))
}

# Define Core and start training
core = PluginCore(datasets=datasets, algorithms=alg, preprocess=preps, inspectors=ispt)
core.save_core(file_dir=file_dir, file_name=start_save_name)

# __________________________________________Model Compare and Inspect_____________________________________________________
cnn = ShallowFBCSPNet(in_chans=8, n_classes=2, input_window_samples=1000, final_conv_length='auto', n_filters_spat=50,n_filters_time=50)
fbcsp = FBCSP(sfreq=250, time_steps=1000, window_start=0, window_length=1000, clf=SVC(probability=True),
              select_ratio=0.6, l_freq=4, h_freq=38, n_cuts=10)
eegnet = EEGNetv4(in_chans=8, n_classes=2, input_window_samples=1000, final_conv_length='auto')

re_subjects_fbcsp, trainned_model_subjects_fbcsp, df_subjects_fbcsp = core.run_cv_subject_dependent(preprocesser_id=2,algorithm_id=2,inspector_id=2,
                                                                                                    n_fold=5,model=fbcsp,trial_start_offset_seconds=0,
                                                                                                    trial_end_offset_seconds=0, direct_window=True,
                                                                                                    model_name='FBCSP', metrics=['acc','kappa','model','subject'])
re_subjects_shallow, trainned_model_subjects_shallow, df_subjects_shallow = core.run_cv_subject_dependent(preprocesser_id=1, algorithm_id=1, inspector_id=1,
                                                                                                          n_fold=5,model=cnn, trial_start_offset_seconds=0,
                                                                                                          trial_end_offset_seconds=0, direct_window=True,
                                                                                                          model_name='ShallowConvNet', metrics=['acc', 'kappa', 'model', 'subject'])
re_subjects_eegnet, trainned_model_subjects_eegnet, df_subjects_eegnet = core.run_cv_subject_dependent(preprocesser_id=1, algorithm_id=1, inspector_id=1,
                                                                                                       n_fold=5,model=eegnet, trial_start_offset_seconds=0,
                                                                                                       trial_end_offset_seconds=0, direct_window=True,
                                                                                                       model_name='EEGNet', metrics=['acc', 'kappa', 'model', 'subject'])

# Inspect
df_compare = df_subjects_eegnet.append([df_subjects_shallow, df_subjects_fbcsp])
sns.boxplot(x='model', y='acc', data=df_compare)
core.log_subject_dependent(df=df_compare, dataset_name=dataset_name, dir=file_dir)

core.save_core(file_dir=file_dir, file_name=compare_save_name)

# __________________________________________Model Search and Inspect_____________________________________________________

params_fbcsp = {
    'sfreq': [250],
    'time_steps': [1000],
    'window_start': [0],
    'window_length': [1000],
    'l_freq': [4],
    'select_ratio': [0.2, 0.4, 0.6, 0.8],
    'h_freq': [20, 30, 40],
    'n_cuts': [5, 10]
}

params_cnn = {
    'module__in_chans': [8],
    'module__n_classes': [2],
    'module__input_window_samples': [1000],
    'module__final_conv_length': ['auto'],
    'module__n_filters_time': [5, 20, 40, 50],
    'module__n_filters_spat': [5, 20, 40, 50],
    'module__drop_prob': [0.2, 0.6, 0.9]
}

train_csp = False
if train_csp:
    fbcsp_search = FBCSPSearch(sfreq=250, time_steps=1000, window_start=0, window_length=1000, select_ratio=0.6,
                               l_freq=4, h_freq=38, n_cuts=10)
    gs_fbcsp = core.search_model(preprocesser_id=2, algorithm_id=5, dataset_id=1, model=fbcsp_search,
                                 params=params_fbcsp, direct_window=True,
                                 subject_mode='subject_dependent', trial_start_offset_seconds=0,
                                 trial_end_offset_seconds=0)

else:
    cnn = ShallowFBCSPNet(in_chans=8, n_classes=2, input_window_samples=1000, final_conv_length='auto',
                          n_filters_spat=50, n_filters_time=50, drop_prob=0.5)
    gs_cnn = core.search_model(preprocesser_id=1, algorithm_id=3, dataset_id=1, model=cnn, params=params_cnn,
                               direct_window=True,
                               subject_mode='subject_dependent', trial_start_offset_seconds=0,
                               trial_end_offset_seconds=0)

# Inspect
if train_csp:
    df_search = return_df_search(gs_fbcsp, ['select_ratio', 'h_freq', 'n_cuts'])
    boxplot_res(df=df_search, interest_keys=['select_ratio'])
    boxplot_res(df=df_search, interest_keys=['h_freq'])
    boxplot_res(df=df_search, interest_keys=['n_cuts'])
    param3d_viz(gs=gs_fbcsp, params=['select_ratio', 'h_freq', 'n_cuts'])
    core.log_model_search(gs=gs_fbcsp, keys=['select_ratio', 'h_freq', 'n_cuts'], dir=file_dir)
else:
    df_search = return_df_search(gs_cnn, ['module__n_filters_time', 'module__n_filters_spat', 'module__drop_prob'])
    boxplot_res(df=df_search, interest_keys=['module__drop_prob'])
    boxplot_res(df=df_search, interest_keys=['module__n_filters_time'])
    boxplot_res(df=df_search, interest_keys=['module__n_filters_spat'])
    param3d_viz(gs=gs_cnn, params=['module__n_filters_time', 'module__n_filters_spat', 'module__drop_prob'])
    core.log_model_search(gs=gs_cnn, keys=['module__n_filters_time', 'module__n_filters_spat', 'module_drop_prob'],
                          dir=file_dir)

core.save_core(file_dir=file_dir, file_name=search_save_name)

# __________________________________________Model Train and Inspect______________________________________________________
cnn = ShallowFBCSPNet(in_chans=8, n_classes=2, input_window_samples=1000, final_conv_length='auto', n_filters_spat=50,
                      n_filters_time=50)
fbcsp = FBCSP(sfreq=250, time_steps=1000, window_start=0, window_length=1000, clf=SVC(probability=True),
              select_ratio=0.6, l_freq=4, h_freq=38, n_cuts=2)

train_csp = True
if train_csp:
    re_subjects_train, trainned_model_subjects_train, df_subjects_train = core.run_cv_subject_dependent(
        preprocesser_id=2, algorithm_id=2, inspector_id=2, n_fold=5,
        model=fbcsp, trial_start_offset_seconds=0, trial_end_offset_seconds=0, direct_window=True,
        model_name='FBCSP', metrics=['acc', 'kappa', 'model', 'subject'])
else:
    re_subjects_train, trainned_model_subjects_train, df_subjects_train = core.run_cv_subject_dependent(
        preprocesser_id=1, algorithm_id=1, inspector_id=1, n_fold=5,
        model=cnn, trial_start_offset_seconds=0, trial_end_offset_seconds=0, direct_window=True,
        model_name='ShallowConvNet', metrics=['acc', 'kappa', 'model', 'subject'])


# Inspect
sns.boxplot(x='model', y='acc', data=df_subjects_train)

i_trained_module = 2
core.modules['trained_module'] = trainned_model_subjects_train[i_trained_module]

core.track_time(preprocesser_id=2, algorithm_id=6, dataset_id=1, trial_start_offset_seconds=0,
                trial_end_offset_seconds=0, direct_window=True,
                model=core.modules['trained_module'], n_inter=10, session_id='1')

# Visualization For FBCSP
if train_csp:
    i_train_fold = 0
    (train_Xs, train_ys), (test_Xs, test_ys) = core.provide_Xys(dataset_id=1, preprocess_id=2,
                                                                subject_mode='subject_dependent',
                                                                train_mode='cross_validation',
                                                                trial_start_offset_seconds=0,
                                                                trial_end_offset_seconds=0, train_r=None, n_fold=5,
                                                                direct_window=True)
    core.algorithms[6].compile(core.modules['trained_module'])
    core.algorithms[6].visualize_train_test_dist(train_X=train_Xs[i_train_fold], train_y=train_ys[i_train_fold],
                                                 test_X=test_Xs[i_train_fold], test_y=test_ys[i_train_fold])
else:
    i_train_fold = 0
    (train_Xs, train_ys), (test_Xs, test_ys) = core.provide_Xys(dataset_id=1, preprocess_id=1,
                                                                subject_mode='subject_dependent',
                                                                train_mode='cross_validation',
                                                                trial_start_offset_seconds=0,
                                                                trial_end_offset_seconds=0, train_r=None, n_fold=5,
                                                                direct_window=True)
    core.algorithms[6].compile(core.modules['trained_module'])
    core.algorithms[6].visualize_train_test_dist(train_X=train_Xs[i_train_fold], train_y=train_ys[i_train_fold],
                                                 test_X=test_Xs[i_train_fold], test_y=test_ys[i_train_fold])

core.save_core(file_dir=file_dir, file_name=train_save_name)

# ___________________________________________ Feature Anaysis & Visulization_____________________________________________
_module = FBCSP(sfreq=250, time_steps=1000, window_start=0, window_length=1000, clf=SVC(probability=True),
                select_ratio=0.6, l_freq=4, h_freq=38, n_cuts=10)

# Evaluate specific time_interval's discriminative power
temporal_res = core.feature_analysis_wrapper(preprocess_id=2, algorithm_id=7, dataset_id=1,
                                             subject_mode='subject_dependent',
                                             train_mode='hold_out', trial_start_offset_seconds=0,
                                             trial_end_offset_seconds=0,
                                             train_r=0.8, n_fold=None, model=_module, direct_window=True,
                                             sub_channels=[list(range(8))],
                                             time_intervals=[(0, 500), (100, 600), (200, 700), (300, 800)])
core.algorithms[7].plot_metric(res=temporal_res, metric='acc')

spat_res = core.feature_analysis_wrapper(preprocess_id=2, algorithm_id=7, dataset_id=1,
                                         subject_mode='subject_dependent',
                                         train_mode='hold_out', trial_start_offset_seconds=0,
                                         trial_end_offset_seconds=0,
                                         train_r=0.8, n_fold=None, model=_module,
                                         sub_channels=[[i] for i in list(range(8))],
                                         direct_window=True, time_intervals=[(0, 1000)])

epoch_info = core.datasets[1].windows[0].info
info = mne.create_info(ch_names=epoch_info['ch_names'], sfreq=1, ch_types=['eeg'] * len(epoch_info['ch_names']))
info.set_montage('standard_1020')
core.algorithms[7].draw_topomap(spat_res, info)



