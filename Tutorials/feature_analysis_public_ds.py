from PluginCore.Datasets.moabb import MOABBDataset,MOABBInfo
from Paradigm.MI.utils import draw_p_test_EpochsTF,get_EpochsTF,draw_band_EpochsTF
import mne

i_moabb_dataset = 1
i_subject = 2
i_session = 'session_T'

# Define datasets
dataset_name, n_subject, _ = MOABBInfo().get_info(i_moabb_dataset)
datasets = {}
datasets[i_subject] = MOABBDataset(dataset_name=dataset_name,subject_ids=[i_subject])

sessions = datasets[i_subject].split('session')
session = sessions[i_session]

raw = mne.concatenate_raws([ds.raw for ds in session.datasets])
events, event_ids = mne.events_from_annotations(raw)
tmin, tmax = -2, 4
want_chs = ('C3', 'Cz', 'C4')

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=want_chs, baseline=None, preload=True)

tfr, df = get_EpochsTF(epochs=epochs, want_chs=want_chs, tmin=tmin, tmax=tmax)
draw_band_EpochsTF(tfr_df=df, want_chs=want_chs)
draw_p_test_EpochsTF(tfr=tfr, want_chs=want_chs,    event_ids=event_ids)


