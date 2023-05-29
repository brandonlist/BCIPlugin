import pandas as pd
import mne
from copy import deepcopy

from PluginCore.Datasets.base import BaseDataset,BaseConcatDataset
from manifest import mne_path

moabb_dataset_list = [('AlexMI',8,'MI'),
                 ('BNCI2014001',9,'MI'),
                 ('BNCI2014002',14,'MI'),
                 ('BNCI2014004',9,'MI'),
                 ('BNCI2015001',12,'MI'),
                 ('BNCI2015004',9,'MI'),
                 ('Cho2017',52,'MI'),
                 ('Shin2017A',29,'MI'),
                 ('Shin2017B',29,'MI'),
                 ('Weibo2014',10,'MI'),

                 ('SSVEPExo',12,'SSVEP'),
                 ('Nakanishi2015',9,'SSVEP'),
                 ('Wang2016',34,'SSVEP'),
                 ('MAMEM1',10,'SSVEP'),
                 ('MAMEM2',10,'SSVEP'),
                 ('MAMEM3',10,'SSVEP'),

                 ('bi2013a',24,'P300'),
                 ('BNCI2014008',8,'P300'),
                 ('BNCI2014009',10,'P300'),
                 ('BNCI2015003',10,'P300')
                 ]

class MOABBInfo():
    def __init__(self):
        pass

    @staticmethod
    def get_info(index):
        info = moabb_dataset_list[index]
        return (info[0], info[1], info[2])


def _find_dataset_in_moabb(dataset_name):
    # soft dependency on moabb
    from moabb.datasets.utils import dataset_list
    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            # return an instance of the found dataset class
            return dataset()
    raise ValueError("'dataset_name' not found in moabb datasets")

def _fetch_and_unpack_moabb_data(dataset, subject_ids, path=mne_path,verbose=True):
    if subject_ids==None:
        subject_ids = dataset.subject_list
    # data = {sub_i:dataset.data_path(subject=sub_i, path=path) for sub_i in subject_ids}
    data = dataset.get_data(subject_ids)

    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id, subj_data in data.items():
        if verbose:
            print('reading from subject',subj_id,'; number of sessions:',len(subj_data))
        for sess_id, sess_data in subj_data.items():
            if verbose:
                print('number of runs for session ',sess_id,':',len(sess_data))
            for run_id, raw in sess_data.items():
                # set annotation if empty
                if len(raw.annotations) == 0:
                    annots = _annotations_from_moabb_stim_channel(raw, dataset)
                    raw.set_annotations(annots)
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    if verbose:
        print('event_ids of dataset:',dataset.event_id)
    description = pd.DataFrame({
        'subject': subject_ids,
        'session': session_ids,
        'run': run_ids
    })
    return raws, description

def _annotations_from_moabb_stim_channel(raw, dataset):
    # find events from stim channel
    events = mne.find_events(raw,verbose=False)

    # get annotations from events
    event_desc = {k: v for v, k in dataset.event_id.items()}
    annots = mne.annotations_from_events(events, raw.info['sfreq'], event_desc)

    # set trial on and offset given by moabb
    onset, offset = dataset.interval
    annots.onset += onset
    annots.duration += offset - onset
    return annots

def fetch_data_with_moabb(dataset_name, subject_ids):
    # ToDo: update path to where moabb downloads / looks for the data
    """Fetch data using moabb.

    Parameters
    ----------
    dataset_name: str
        the name of a dataset included in moabb
    subject_ids: list(int) | int
        (list of) int of subject(s) to be fetched

    Returns
    -------
    raws: mne.Raw
    info: pandas.DataFrame
    """
    dataset = _find_dataset_in_moabb(dataset_name)
    subject_id = [subject_ids] if isinstance(subject_ids, int) else subject_ids
    return _fetch_and_unpack_moabb_data(dataset, subject_id)

def get_raw_rest_stated(raw):
    class_onsets = [anno['onset'] for anno in raw.annotations]
    class_duration = [anno['duration'] for anno in raw.annotations]
    class_description = [anno['description'] for anno in raw.annotations]

    for i in range(len(class_onsets) - 1):
        raw.annotations.append(onset=class_onsets[i] + class_duration[i],
                               duration=class_onsets[i + 1] - (class_onsets[i] + class_duration[i]),
                               description='rest_state')

    class_indexs = raw.annotations.description != 'rest_state'
    raw.annotations.delete(class_indexs)
    return raw

def get_dataset_rest_stated(dataset_orig):
    dataset = deepcopy(dataset_orig)
    for ds_i,ds in enumerate(dataset.datasets):
        dataset.datasets[ds_i].raw = get_raw_rest_stated(ds.raw)
    return dataset

class MOABBDataset(BaseConcatDataset):
    """A class for moabb datasets.

    Parameters
    ----------
    dataset_name: name of dataset included in moabb to be fetched
    subject_ids: list(int) | int
        (list of) int of subject(s) to be fetched
    """
    def __init__(self, dataset_name, subject_ids):
        raws, description = fetch_data_with_moabb(dataset_name, subject_ids)
        all_base_ds = [BaseDataset(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        self.dataset_name = dataset_name
        super().__init__(all_base_ds)
