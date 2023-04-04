import os.path
import pickle
import time
from datetime import datetime

from tqdm import tqdm
import mne
import numpy as np
import pandas as pd
from copy import deepcopy

from PluginCore.Datasets.base import fetch_data_description, WindowsDataset
from Paradigm.base import SynParadigm,Result
from bGUI.pluginWindow import WinMICalibrate

def load_MITrials_from_raw(raw, event_id, trial_start_offset_seconds, trial_length_seconds):
    # TODO: change to using the functionns in windowers, since we need a uniform API
    # event_id = {'48': 0, '49': 1}
    MITrials = []
    for an in raw.annotations:
        if an['description'] in event_id.keys():
            start = int((an['onset'] + trial_start_offset_seconds) * raw.info['sfreq'])
            stop = start + int(trial_length_seconds * raw.info['sfreq'])
            data = raw.get_data(start=start, stop=stop)
            target = event_id[an['description']]

            MITrials.append((data, target))
    return MITrials


class MICalibrateParadigm(SynParadigm):
    def __init__(self, BCIServer, config=None, log_func=print):
        SynParadigm.__init__(self, BCIServer=BCIServer)
        self.init_config(config)
        self.running_param = {
            'i_session':-1,
            'i_run':-1,
            'i_trial':-1,
            'is_Listening':False,
            'stream_id':-1,
        }
        self.log_func = log_func
        self.MITrials = []
        self.ConfigWindow = None

    def configParadigm(self):
        self.ConfigWindow = WinMICalibrate(paradigm=self)
        self.ConfigWindow.show()

        self.ConfigWindow = None

    def SendMItype(self, name):
        option = 'On' if self.config[name] else 'Off'
        cmd = 'Cmd_MICalibrate_Set' + name + option
        self.BCIServer.broadcastCmd(cmd)

    def SendConfig(self):
        delay = 1
        pbar = tqdm(total=10)

        self.BCIServer.broadcastCmd('Cmd_MICalibrate_SetNSession_'+str(self.config['n_session']))
        time.sleep(delay)
        pbar.update(1)

        self.BCIServer.broadcastCmd('Cmd_MICalibrate_SetNRun_'+str(self.config['n_run']))
        time.sleep(delay)
        pbar.update(1)

        self.BCIServer.broadcastCmd('Cmd_MICalibrate_SetNTrial_'+str(self.config['n_trial']))
        time.sleep(delay)
        pbar.update(1)

        self.BCIServer.broadcastCmd('Cmd_MICalibrate_SetTrialLength_'+str(self.config['TrialLength']))
        time.sleep(delay)
        pbar.update(1)

        self.SendMItype('LeftHand')
        time.sleep(delay)
        pbar.update(1)

        self.SendMItype('RightHand')
        time.sleep(delay)
        pbar.update(1)

        self.SendMItype('Rest')
        time.sleep(delay)
        pbar.update(1)

        self.SendMItype('LeftFoot')
        time.sleep(delay)
        pbar.update(1)

        self.SendMItype('RightFoot')
        time.sleep(delay)
        pbar.update(1)

        self.SendMItype('Tongue')
        time.sleep(delay)
        pbar.update(1)

        pbar.close()

    def run(self):
        self.reset()
        self.SendConfig()
        self.startListening()
        self.BCIServer.broadcastCmd('Cmd_MICalibrate_Start')

    def stop(self):
        self.stopListening()

    def init_config(self, config):
        default_config = {
            'n_session':-1,
            'n_run':-1,
            'n_trial':-1,
            'DataPeriod':-1,
            'TrialLength':-1,
            'LeftHand':False,
            'RightHand':False,
            'Rest':False,
            'LeftFoot':False,
            'RightFoot':False,
            'Tongue':False
        }
        if config is None:
            self.config = default_config
        else:
            for k in default_config:
                self.config[k] = config[k]

    def reset(self):
        self.running_param['i_session'] = 0
        self.running_param['i_run'] = 0
        self.running_param['i_trial'] = 0
        self.MITrials = []

    def createDataset(self, trial_start, trial_length, subject_id, info):
        arr = np.stack([t for t, _ in self.MITrials])
        labels = np.array([l for _,l in self.MITrials],dtype=np.int)
        events = np.zeros((len(self.MITrials), 3), dtype=np.int)

        classes = np.unique(labels)
        class_map = {}
        for i,c in enumerate(classes):
            class_map[c] = i

        for i,l in enumerate(labels):
            labels[i] = class_map[l]

        events[:, 1] = trial_length
        events[:, 2] = labels
        events[:, 0] = np.arange(0, len(self.MITrials))

        metadata = pd.DataFrame({
            'i_window_in_trial': np.zeros((len(events)), dtype=np.int),
            'i_start_in_trial': np.array([trial_start] * len(events)),
            'i_stop_in_trial': np.array([trial_start + trial_length] * len(events)),
            'target': events[:, 2]
        })
        mne_epochs = mne.EpochsArray(data=arr, info=info, events=events,
                                     metadata=metadata)
        data = {subject_id: {'session 0': {'run 0': mne_epochs}}}
        description = fetch_data_description(data)[1]

        windowedDataset = WindowsDataset(windows=mne_epochs, description=description)
        return windowedDataset, class_map

    def save_calibrate_to_pkl_file(self, windowedDataset, class_map, filepath, filename=''):
        # time_info = datetime.now().strftime("%Y-%m-%d-%H-%M")
        pth = os.path.join(filepath, filename + '.pkl')
        data = {}
        data['windowedDataset'] = windowedDataset
        data['class_map'] = class_map

        inverse_class_map = {}
        for k in class_map:
            v = class_map[k]
            inverse_class_map[v] = k
        data['inverse_class_map'] = inverse_class_map

        with open(pth, 'wb') as f:
            pickle.dump(data,f)

    def EventHandler(self, type):
        if type=='session start':
            self.running_param['i_session'] += 1
            self.running_param['i_run'] = -1
            self.running_param['i_trial'] = -1
            self.log_func('Session '+str(self.running_param['i_session'])+' Start')

        if type=='session end':
            self.log_func('Session '+str(self.running_param['i_session']+1)+' End')

        if type=='run start':
            self.running_param['i_run'] += 1
            self.running_param['i_trial'] = -1
            self.log_func('Run '+str(self.running_param['i_run']+1)+' Start')

        if type=='run end':
            self.log_func('Run '+str(self.running_param['i_run']+1)+' End')

        if type=='trial start':
            self.running_param['i_trial'] += 1
            self.log_func('Trial '+str(self.running_param['i_trial']+1)+' Start')

        if type=='trial end':
            self.log_func('Trial '+str(self.running_param['i_trial']+1)+' End')

        if type=='record data':
            self.log_func('MI Calibration Paradigm: Recording data...')

            stream_id = self.running_param['stream_id']
            dataPeriod = self.config['DataPeriod']
            data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=dataPeriod)
            y = self.BCIServer.valueService.values['MIstate']
            self.MITrials.append((deepcopy(data),y))

    def startListening(self):
        self.BCIServer.eventService.typeChangedHandler.update({'MICalibrate': self.EventHandler})
        self.running_param['is_Listening'] = True

    def stopListening(self):
        if self.running_param['is_Listening']:
            self.BCIServer.eventService.typeChangedHandler.pop('MICalibrate')
            self.running_param['is_Listening'] = False


