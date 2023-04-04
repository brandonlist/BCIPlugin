import mne
from sklearn.svm import SVC
import seaborn as sns

from PluginCore.Decoder.module.CSP.FBCSP import FBCSP,FBCSPSearch
from PluginCore.Decoder.trainer.utils import plot_res, boxplot_res, param3d_viz, return_df_search
from PluginCore.Datasets.moabb import MOABBDataset,MOABBInfo
from PluginCore.Processor.base import Processor
from PluginCore.Processor.module.temp import exponential_moving_standardize,convert_to_uV
from PluginCore.Decoder.trainer.Skorch import SkorchFactory,SkorchRandSearch,SkorchGridSearch
from PluginCore.Decoder.trainer.csp import CSPFactory,CSPRandomSearch,CSPFilter,CSPWrapper
from PluginCore.Decoder.inspector.csp import CSPInspector
from PluginCore.Decoder.inspector.Skorch import SkorchInspector
from PluginCore.Decoder.inspector.base import InspectorSyn
from PluginCore.Decoder.ml import classifibility
from PluginCore.base import PluginCore
from PluginCore.Decoder.module.braindecode.shallow_fbcsp import ShallowFBCSPNet
from PluginCore.Decoder.module.braindecode.eegnet import EEGNetv4


i_moabb_dataset = 1
i_subject = 1

file_dir = '.\\Tutorials\\data\\Train_Model_public_ds\\'
start_savename = 'CoreModelStart.pkl'
compare_savename = 'CoreModelCompare.pkl'
train_savename = 'CoreModelTrain.pkl'


# Define datasets
dataset_name, n_subject, _ = MOABBInfo().get_info(i_moabb_dataset)
datasets = {}
datasets[i_subject] = MOABBDataset(dataset_name=dataset_name,subject_ids=[i_subject])

# Define Preprocecss
low_cut_hz = 4.
high_cut_hz = 38.
factor_new = 1e-3
init_block_size = 1000


preps = {
    1:[Processor('pick',picks='eeg'),
       Processor(convert_to_uV),  # Convert from V to uV
       Processor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
       Processor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size),
       Processor('resample',sfreq=50)
       ],
    2:[Processor('pick',picks='eeg')]
}

# Define algorithm
alg = {
    1: SkorchFactory(Model=ShallowFBCSPNet,max_epochs=600),
    2: CSPFactory(Model=FBCSP),
    3: SkorchRandSearch(Model=ShallowFBCSPNet,n_iter=10,max_epochs=250),
    4: SkorchGridSearch(Model=ShallowFBCSPNet),
    5: CSPRandomSearch(Model=FBCSP,n_iter=10),
    6: CSPFilter(filter_func=classifibility,model=None),
    7: CSPWrapper(train_factory=CSPFactory(Model=FBCSP),inspector_factory=CSPInspector(InspectorSyn(pos_label=None))),
}

# Define inspector
ispt = {
    1:SkorchInspector(InspectorSyn(pos_label=None),cuda=True),
    2:CSPInspector(InspectorSyn(pos_label=None))
}


# Define Core and start training
core = PluginCore(datasets=datasets,algorithms=alg,preprocess=preps,inspectors=ispt)
core.save_core(file_dir=file_dir,file_name=start_savename)


#___________________________________________Model Compare and Inspect___________________________________________________
cnn = ShallowFBCSPNet(in_chans=22,n_classes=4,input_window_samples=200,final_conv_length='auto',n_filters_spat=50,n_filters_time=50)
fbcsp = FBCSP(sfreq=250,time_steps=1000,window_start=0,window_length=1000,clf=SVC(probability=True),select_ratio=0.6,l_freq=4,h_freq=38,n_cuts=10)
eegnet = EEGNetv4(in_chans=22,n_classes=4,input_window_samples=200,final_conv_length='auto')


re_subjects_fbcsp, trainned_model_subjects_fbcsp, df_subjects_fbcsp = core.run_cv_subject_dependent(preprocesser_id=2,algorithm_id=2,inspector_id=2,n_fold=5,
                                                                           model=fbcsp,trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                                                                model_name='FBCSP',metrics=['acc','kappa','model','subject'])
re_subjects_shallow, trainned_model_subjects_shallow, df_subjects_shallow = core.run_cv_subject_dependent(preprocesser_id=1,algorithm_id=1,inspector_id=1,n_fold=5,
                                                                           model=cnn,trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                                                                model_name='ShallowConvNet',metrics=['acc','kappa','model','subject'])
re_subjects_eegnet, trainned_model_subjects_eegnet, df_subjects_eegnet = core.run_cv_subject_dependent(preprocesser_id=1,algorithm_id=1,inspector_id=1,n_fold=5,
                                                                           model=eegnet,trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                                                                model_name='EEGNet',metrics=['acc','kappa','model','subject'])

#Inspect
df_compare = df_subjects_eegnet.append([df_subjects_shallow, df_subjects_fbcsp])
sns.boxplot(x='model',y='acc',data=df_compare)
core.log_subject_dependent(df=df_compare,dataset_name=dataset_name,dir=file_dir)
core.save_core(file_dir=file_dir,file_name=compare_savename)




#______________________________________________Model Search and Inspect_________________________________________________
params_fbcsp = {
    'sfreq':[250],
    'time_steps':[1000],
    'window_start':[0],
    'window_length':[1000],
    'l_freq':[4],
    'select_ratio':[0.2, 0.4, 0.6, 0.8],
    'h_freq':[20, 30, 40],
    'n_cuts':[5,10]
}


params_cnn = {
    'module__in_chans':[22],
    'module__n_classes':[4],
    'module__input_window_samples':[200],
    'module__final_conv_length':['auto'],
    'module__n_filters_time':[20,40,60],
    'module__n_filters_spat':[20,40,60],
    'module__drop_prob':[0.2,0.4,0.6]
}

train_csp = True
if train_csp:
    fbcsp_search = FBCSPSearch(sfreq=250,time_steps=1000,window_start=0,window_length=1000,select_ratio=0.6,l_freq=4,h_freq=38,n_cuts=10)
    gs_fbcsp = core.search_model(preprocesser_id=2,algorithm_id=5,dataset_id=1,model=fbcsp_search,params=params_fbcsp,
                           subject_mode='subject_dependent',trial_start_offset_seconds=0,trial_end_offset_seconds=0)

else:
    cnn = ShallowFBCSPNet(in_chans=22,n_classes=4,input_window_samples=200,final_conv_length='auto',n_filters_spat=50,n_filters_time=50)
    gs_cnn = core.search_model(preprocesser_id=1,algorithm_id=3,dataset_id=1,model=cnn,params=params_cnn,
                           subject_mode='subject_dependent',trial_start_offset_seconds=0,trial_end_offset_seconds=0)

#Inspect
if train_csp:
    df_search = return_df_search(gs_fbcsp, ['select_ratio','h_freq','n_cuts'])
    boxplot_res(df=df_search, interest_keys=['select_ratio'])
    boxplot_res(df=df_search, interest_keys=['h_freq'])
    boxplot_res(df=df_search, interest_keys=['n_cuts'])
    param3d_viz(gs=gs_fbcsp, params=['select_ratio','h_freq','n_cuts'])
    core.log_model_search(gs=gs_fbcsp,  keys=['select_ratio','h_freq','n_cuts'], dir=file_dir)
else:
    df_search = return_df_search(gs_cnn, ['module__n_filters_time','module__n_filters_spat','module_drop_prob'])
    boxplot_res(df=df_search, interest_keys=['module_drop_prob'])
    boxplot_res(df=df_search, interest_keys=['module__n_filters_time'])
    boxplot_res(df=df_search, interest_keys=['module__n_filters_spat'])
    param3d_viz(gs=gs_cnn, params=['module__n_filters_time','module__n_filters_spat','module_drop_prob'])
    core.log_model_search(gs=gs_cnn, keys=['module__n_filters_time','module__n_filters_spat','module_drop_prob'], dir=file_dir)




#________________________________________________Model Train and Inspect________________________________________________
fbcsp = FBCSP(sfreq=250,time_steps=1000,window_start=0,window_length=1000,clf=SVC(probability=True),select_ratio=0.8,l_freq=4,h_freq=30,n_cuts=10)
re_subjects_fbcsp_train, trainned_model_subjects_fbcsp_train, df_subjects_fbcsp_train = core.run_cv_subject_dependent(preprocesser_id=2,algorithm_id=2,inspector_id=2,n_fold=5,
                                                                           model=fbcsp,trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                                                                model_name='FBCSP',metrics=['acc','kappa','model','subject'])

#Inspect
sns.boxplot(x='model',y='acc',data=df_subjects_fbcsp_train)

i_trained_module = 1
core.modules['trained_module'] = trainned_model_subjects_fbcsp_train[i_trained_module]

core.track_time(preprocesser_id=2,algorithm_id=6,dataset_id=1,trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                model=core.modules['trained_module'],n_inter=4, session_id='session_E')


i_train_fold = 0
(train_Xs, train_ys),(test_Xs, test_ys) = core.provide_Xys(dataset_id=1,preprocess_id=2,subject_mode='subject_dependent',train_mode='cross_validation',
                       trial_start_offset_seconds=0,trial_end_offset_seconds=0,train_r=None,n_fold=5)
core.algorithms[6].compile(core.modules['trained_module'])
core.algorithms[6].visualize_train_test_dist(train_X=train_Xs[i_train_fold],train_y=train_ys[i_train_fold],
                                             test_X=test_Xs[i_train_fold],test_y=test_ys[i_train_fold])


core.save_core(file_dir=file_dir,file_name=train_savename)



#___________________________________________ Feature Anaysis & Visulization_____________________________________________
_module =  FBCSP(sfreq=250,time_steps=1000,window_start=0,window_length=1000,clf=SVC(probability=True),select_ratio=0.8,l_freq=4,h_freq=30,n_cuts=10)

# Evaluate specific time_interval's discriminative power
temporal_res = core.feature_analysis_wrapper(preprocess_id=2,algorithm_id=7,dataset_id=1,subject_mode='subject_dependent',
                                    train_mode='hold_out',trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                                    train_r=0.8,n_fold=None,model=_module,
                                    sub_channels=[list(range(22))],
                                    time_intervals=[(0,500),(100,600),(200,700),(300,800)])
core.algorithms[7].plot_metric(res=temporal_res, metric='acc')



spat_res = core.feature_analysis_wrapper(preprocess_id=2,algorithm_id=7,dataset_id=1,subject_mode='subject_dependent',
                                    train_mode='hold_out',trial_start_offset_seconds=0,trial_end_offset_seconds=0,
                                    train_r=0.8,n_fold=None,model=_module,
                                    sub_channels=[[i] for i in list(range(22))],
                                    time_intervals=[(0,1000)])

raw = core.extract_raw_moabb(dataset_id=1, raw_id=1, preprocesser_id=2)
info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=1, ch_types=['eeg'] * len(raw.info['ch_names']))
info.set_montage('standard_1020')
core.algorithms[7].draw_topomap(spat_res, info)


