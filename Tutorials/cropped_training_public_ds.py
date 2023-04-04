import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import numpy as np
from sklearn.metrics import confusion_matrix

from PluginCore.Decoder.module.braindecode.shallow_fbcsp import ShallowFBCSPNet
from PluginCore.Datasets.moabb import MOABBDataset
from PluginCore.Processor.base import preprocess,Processor
from PluginCore.Processor.module.temp import exponential_moving_standardize
from PluginCore.Decoder.module.braindecode.core.util import to_dense_prediction_model, get_output_shape
from PluginCore.Datasets.utils.windowers import create_windows_from_events

subject_id = 2
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

low_cut_hz = 4.
high_cut_hz = 38.
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Processor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Processor(lambda x: x * 1e6),  # Convert from V to uV
    Processor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Processor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

preprocess(dataset, preprocessors)

input_window_samples = 1000
n_classes = 4
n_chans = dataset[0][0].shape[0]

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True


model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length=30,
)

# Send model to GPU
if cuda:
    model.cuda()

to_dense_prediction_model(model)
n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]



trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True
)

splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']


from PluginCore.Decoder.module.braindecode.core.classifier import EEGClassifier
from PluginCore.Decoder.module.braindecode.core.losses import CroppedLoss

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 400

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)



y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

confusion_mat = confusion_matrix(y_true, y_pred)
acc = np.trace(confusion_mat)/np.sum(confusion_mat)
