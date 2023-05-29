import mne
from collections import OrderedDict
import numpy as np
import os
import requests
import pandas as pd
from scipy.io import loadmat

from PluginCore.Datasets.base import BaseConcatDataset,BaseDataset
from manifest import BCIC_dir

def get_raw_trial_from_gdf_file(dataset, filename, classes=None, special_case=None):
    raw_t = mne.io.read_raw_gdf(filename, stim_channel='auto')
    data = raw_t.get_data()
    gdf_events = mne.events_from_annotations(raw_t)
    info = mne.create_info(dataset.channel_names, dataset.fs, dataset.channel_types)
    info.set_montage(dataset.montage)
    raw = mne.io.RawArray(data, info, verbose="WARNING")
    raw.info["gdf_events"] = gdf_events
    for a_t in raw_t.annotations:
        a = OrderedDict()
        for key in a_t:
            a[key] = a_t[key]
        raw.annotations.append(onset=a['onset'], duration=a['duration'], description=a['description'])

    # extract events
    events, name_to_code = raw.info["gdf_events"]
    # name_to_code:{'1023': 1,'1072': 2,'276': 3,'277': 4,'32766': 5, '768': 6,'769': 7,'770': 8,'771': 9,'772': 10}
    if dataset.name == 'BCIC4_2a':
        # There is file in that dataset that lacks 2 EOG signal
        if len(name_to_code) == 8:  # The one file that lacks 2 EOG signal
            trial_codes = [5, 6, 7, 8]
            num_to_detract = 5
            code_start = 4
        elif '783' in name_to_code.keys():  # Reading label file
            trial_codes = [7]
            num_to_detract = 0
            code_start = 6
        else: #normal occasion for that dataset
            trial_codes = [name_to_code[d] for d in dataset.eventDescription if 'Onset' in dataset.eventDescription[d]]
    elif dataset.name == 'BCIC4_2b':
        if '783' in name_to_code.keys():  # Reading label file
            trial_codes = [6,7]
            num_to_detract = 0
            code_start = 9
        else: #normal occasion for that dataset
            trial_codes = [name_to_code[d] for d in dataset.eventDescription if 'Onset' in dataset.eventDescription[d]]
            code_start = 9
    elif dataset.name == 'BCIC3_3b':
        trial_codes = [2,3]
    else:
        trial_codes = [name_to_code[d] for d in dataset.eventDescription if 'Onset' in dataset.eventDescription[d]]

    if dataset.name == 'BCIC3_3a' and classes is not None:
        classes = classes + 2
        j=0
        for i in range(len(events)):
            if events[i,2] == 7 or events[i,2] == 3 or events[i,2]==4 or events[i,2]==5 or events[i,2]==6:
                events[i,2] = classes[j]
                j += 1
            if events[i,2] == 3 or events[i,2]==4 or events[i,2]==5 or events[i,2]==6:
                events[i,2] == np.nan
        events = events[events[:,2]!=np.nan]

    if dataset.name == 'BCIC3_3b' and classes is not None:
        if special_case==None:
            classes = classes + 1
            j = 0
            for i in range(len(events)):
                if events[i,2] == 5 or events[i,2] == 2 or events[i,2]==3:
                    events[i,2] = classes[j]
                    j += 1
                if events[i,2] == 2 or events[i,2]==3:
                    events[i,2] == np.nan
            events = events[events[:,2]!=np.nan]
        elif special_case==1:
            classes = classes + 1
            j = 0
            for i in range(len(events)):
                if events[i, 2] == 2 or events[i, 2] == 3:
                    events[i, 2] == np.nan
            events = events[events[:, 2] != np.nan]
            for i in range(len(events)):
                if events[i, 2] == 5:
                    events[i, 2] = classes[j]
                    j += 1

    if dataset.name == 'BCIC4_2b' and classes is not None:
        if special_case==1:
            classes = classes + 5
            j = 0
            for i in range(len(events)):
                if events[i, 2] == 5:
                    events[i, 2] = classes[j]
                    j += 1
        else:
        # classes = classes + 5
        # j=0
        # for i in range(len(events)):
        #     if events[i,2] == 11 or events[i,2] == 6 or events[i,2]==7:
        #         events[i,2] = classes[j]
        #         j += 1
        #     if events[i,2] == 6 or events[i,2]==7:
        #         events[i,2] == np.nan
        # events = events[events[:,2]!=np.nan]
            events = np.array(events,dtype=np.float)
            classes = classes + 5
            j = 0
            for i in range(len(events)):
                if events[i, 2] == 6.0 or events[i, 2] == 7.0:
                    events[i, 2] = np.nan
            events = events[np.logical_not(np.isnan(events[:,2]))]
            for i in range(len(events)):
                if events[i, 2] == 11.0:
                    events[i, 2] = classes[j]
                    j += 1
            events = np.array(events,dtype=np.int)

    num = trial_codes
    num.sort()
    num_to_detract = num[0]
    del num
    code_start = name_to_code['768']

    trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
    trial_events = events[trial_mask]

    trial_events[:, 2] = trial_events[:, 2] - num_to_detract

    if dataset.name == 'BCIC4_2a':
        # cause only read label from that dataset now
        if '783' in name_to_code.keys():  # meaning is test file
            trial_events[:, 2] = classes - 7
            classes_set = [0, 1, 2, 3]

    raw.info['events'] = trial_events
    # unique_classes = np.unique(trial_events[:, 2])

    if dataset.name=='BCIC3_3b' or dataset.name=='BCIC4_2b':
        #no rejected trials in that dataset file
        return raw

    # now also create 0-1 vector for rejected trials
    if dataset.name == 'BCIC4_2a' or dataset.name == 'BCIC4_2b':
        trial_start_events = events[events[:, 2] == code_start]
    if dataset.name == 'BCIC3_3a':
        # TODO:have bug reading it, not a big problem since not every dataset has mask for rejected trials
        return raw
    #     events_no1 = events[events[:, 2] != 1]
    #     t_time = [events_no1[i + 3, 0] for i, c in enumerate(events_no1[:-3, 2]) if
    #               c == 2 and (events_no1[i + 3, 2] in [3, 4, 5, 6])]
    #     mask = [(t in t_time) for t in events[:,0]]
    #     trial_start_events = events[mask]
    assert len(trial_start_events)==len(trial_events),print(len(trial_start_events),len(trial_events))
    artifact_trial_mask = np.zeros(len(trial_start_events), dtype=np.uint8)
    artifact_events = events[events[:, 2] == name_to_code['1023']]
    for artifact_time in artifact_events[:, 0]:
        try:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
        except:
            continue
        artifact_trial_mask[i_trial] = 1

    # get trial events and artifact_mask

    raw.info['artifact_trial_mask'] = artifact_trial_mask

    # drop EOG channels
    if dataset.name == 'BCIC4_2a':
        raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    if dataset.name == 'BCIC4_2b':
        raw.drop_channels(['EOG:01', 'EOG:02', 'EOG:03'])
    # for events in raw, there are only 4 task events 0,1,2,3
    return raw

def fetch_data_with_BCIC(data):
    raws, subject_ids, session_ids, run_ids = [], [], [], []
    for subj_id,subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id,raw in sess_data.items():
                raws.append(raw)
                subject_ids.append(subj_id)
                session_ids.append(sess_id)
                run_ids.append(run_id)
    description = pd.DataFrame({
            'subject': subject_ids,
            'session': session_ids,
            'run': run_ids
        })
    return raws,description

class BCICompetition3_3a(BaseConcatDataset):
    def __init__(self):
        self.channel_names = ['AFFz', 'F1h', 'Fz', 'F2h', 'FFC1', 'FFC1h', 'FFCz', 'FFC2h', 'FFC2', 'FC3h', 'FC1',
                              'FC1h', 'FCz',
                              'FC2h', 'FC2', 'FC4h', 'FCC3', 'FCC3h', 'FCC1', 'FCC1h', 'FCCz', 'FCC2h', 'FCC2', 'FCC4h',
                              'FCC4',
                              'C5', 'C5h', 'C3', 'C3h', 'C1h', 'Cz', 'C2h', 'C4h', 'C4', 'C6h', 'C6', 'CCP3', 'CCP3h',
                              'CCP1', 'CCP1h',
                              'CCPz', 'CCP2h', 'CCP2', 'CCP4h', 'CCP4', 'CP3h', 'CP1', 'CP1h', 'CPz', 'CP2h', 'CP2',
                              'CP4h', 'CPP1',
                              'CPP1h', 'CPPz', 'CPP2h', 'CPP2', 'P1h', 'Pz', 'P2h']
        self.fs = 250
        self.channel_types = ['eeg'] * 60
        self.montage = 'standard_1005'
        self.name = 'BCIC3_3a'
        self.n_subject = 3
        self.eventDescription = {'768': 'startTrail',
                                 '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight',
                                 '771': 'cueOnsetFoot',
                                 '772': 'cueOnsetTongue',
                                 '783': 'cueUnknown',
                                 '1023': 'rejectedTrial',
                                 '785': 'beep',
                                 '786': 'crossOnScreen'}

        label_file_names = ['true_label_k3.txt', 'true_label_k6.txt', 'true_label_l1.txt']
        label_html_url = r'http://www.bbci.de/competition/iii/results/graz_IIIa/'

        data = {}
        for i_sub in range(self.n_subject):
            data['subject'+str(i_sub)] = {}
        path=os.path.join(BCIC_dir,r'3_3a')
        for i in range(self.n_subject):
            path_s = os.path.join(path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s, file_path)

                    html = requests.get(os.path.join(label_html_url, label_file_names[i]))
                    classes = np.array([int(i) for i in html.content.decode().split('\n')[:-1]])

                    raw_train = get_raw_trial_from_gdf_file(self, path_s)
                    raw_test = get_raw_trial_from_gdf_file(self, path_s, classes)

                    classes_idx = [i for i, an in enumerate(raw_train.annotations) if
                                   an['description'] not in ['769', '770', '771', '772']]
                    raw_train.annotations.delete(classes_idx)

                    classes_idx = [i for i, an in enumerate(raw_test.annotations) if
                                   an['description'] not in ['769', '770', '771', '772']]
                    raw_test.annotations.delete(classes_idx)

                    data['subject'+str(i)] = {
                        'session_T':{'run_0':raw_train},
                        'session_E':{'run_0':raw_test}
                    }

        raws, description = fetch_data_with_BCIC(data)

        all_base_ds = [BaseDataset(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        super(BCICompetition3_3a, self).__init__(all_base_ds)

class BCICompetition3_3b(BaseConcatDataset):
    def __init__(self):
        #Subject 1 has problem
        self.channel_names = ['C3','C4']
        self.fs = 125
        self.channel_types = ['eeg'] * 2
        self.montage = 'standard_1005'
        self.name = 'BCIC3_3b'
        self.n_subject = 2
        self.eventDescription = {'768': 'startTrail',
                                 '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight',
                                 '781':'feedbackOnsetContinuous',
                                 '783': 'cueUnknown',
                                 '785': 'beep'}
        label_file_names = ['true_labels_O3VR.txt', 'true_labels_S4.txt', 'true_labels_X11.txt']
        label_html_url = r'http://www.bbci.de/competition/iii/results/graz_IIIb/'

        data = {}
        for i_sub in range(self.n_subject):
            data['subject' + str(i_sub)] = {}
        path = os.path.join(BCIC_dir, r'3_3b')
        for i in range(self.n_subject):
            i = i + 1
            path_s = os.path.join(path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s, file_path)

                    html = requests.get(os.path.join(label_html_url, label_file_names[i]))
                    if i == 0:
                        classes = np.array([int(i) for i in html.content.split(b';')[-1].decode().split('\n')[1:-1]])
                    else:
                        classes = np.array([int(i) for i in html.content.decode().split('\n')[:-1]])

                    raw_train = get_raw_trial_from_gdf_file(self, path_s)
                    if i == 0:
                        raw_test = get_raw_trial_from_gdf_file(self, path_s, classes, special_case=1)
                    else:
                        raw_test = get_raw_trial_from_gdf_file(self, path_s, classes)

                    classes_idx = [i for i, an in enumerate(raw_train.annotations) if
                                   an['description'] not in ['769', '770']]
                    raw_train.annotations.delete(classes_idx)

                    classes_idx = [i for i, an in enumerate(raw_test.annotations) if
                                   an['description'] not in ['769', '770']]
                    raw_test.annotations.delete(classes_idx)

                    data['subject' + str(i)] = {
                        'session_T': {'run_0': raw_train},
                        'session_E': {'run_0': raw_test}
                    }

        raws, description = fetch_data_with_BCIC(data)

        all_base_ds = [BaseDataset(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        super(BCICompetition3_3b, self).__init__(all_base_ds)

class BCICompetition3_4a(BaseConcatDataset):
    def __init__(self):
        self.path = os.path.join(BCIC_dir, '3_4a')

        self.montage = 'standard_1005'
        self.name = 'BCIC3_4a'
        self.n_subject = 5
        self.classes = [1, 2]

        data = {}
        for i_sub in range(self.n_subject):
            data['subject'+str(i_sub)] = {}
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path = [p for p in os.listdir(os.path.join(self.path, 's' + str(i + 1))) if
                             (p.split('.')[-1] == 'mat') and (p.split('_')[0] == 'data')][0]
            file_mat_path = os.path.join(self.path, 's' + str(i + 1), file_mat_path)
            data_subject_i = loadmat(file_mat_path)

            file_label_path = [p for p in os.listdir(os.path.join(self.path, 's' + str(i + 1))) if
                               (p.split('.')[-1] == 'mat') and (p.split('_')[0] == 'true')][0]
            file_label_path = os.path.join(self.path, 's' + str(i + 1), file_label_path)
            label_subject_i = loadmat(file_label_path)

            Pos = data_subject_i['mrk']['pos'][0][0][0]
            Y = label_subject_i['true_y'][0]

            classnames = [classname[0] for classname in data_subject_i['mrk']['className'][0][0][0]]
            self.channel_names = [clab[0] for clab in data_subject_i['nfo']['clab'][0][0][0]]
            xpos = data_subject_i['nfo']['xpos'][0][0]
            ypos = data_subject_i['nfo']['ypos'][0][0]
            sfreq = data_subject_i['nfo']['fs'][0][0][0][0]
            raw_data = data_subject_i['cnt'].T * 1e-6

            info = mne.create_info(ch_names=self.channel_names, sfreq=sfreq, ch_types=['eeg'] * 118)
            raw = mne.io.RawArray(data=raw_data, info=info)
            for p, y in zip(Pos, Y):
                raw.annotations.append(onset=p / sfreq, duration=3.5,
                                       description=y)  # as in dataset description: https://bbci.de/competition/iii/desc_IVa.html

            data['subject' + str(i)] = {
                'session_0': {'run_0': raw},
            }

        raws, description = fetch_data_with_BCIC(data)
        self.fs = sfreq

        all_base_ds = [BaseDataset(raw, row)
                       for raw, (_, row) in zip(raws, description.iterrows())]
        super(BCICompetition3_4a, self).__init__(all_base_ds)