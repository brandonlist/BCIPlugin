import numpy as np

from Paradigm.base import SynParadigm
from tqdm import tqdm
import time
from copy import deepcopy

class MIOnlineParadigm(SynParadigm):
    def __init__(self, BCIServer, preprocess, trainned_model, inverse_class_map=None,
                 config=None, log_func=print):
        SynParadigm.__init__(self, BCIServer=BCIServer)
        self.init_config(config)
        self.running_param = {
            'i_session':-1,
            'i_run':-1,
            'i_trial':-1,
            'is_Listening':False,
            'stream_id':-1,
            'client_id':-1,
        }
        self.log_func = log_func
        self.ConfigWindow = None

        self.preprocess = preprocess
        self.trainned_model = trainned_model
        self.inverse_class_map = inverse_class_map

    def configParadigm(self):
        pass

    def SendMItype(self, name):
        option = 'On' if self.config[name] else 'Off'
        cmd = 'Cmd_MIOnline_Set' + name + option
        self.BCIServer.broadcastCmd(cmd)

    def SendConfig(self):
        delay = 1
        pbar = tqdm(total=10)

        self.BCIServer.broadcastCmd('Cmd_MIOnline_SetNSession_'+str(self.config['n_session']))
        time.sleep(delay)
        pbar.update(1)

        self.BCIServer.broadcastCmd('Cmd_MIOnline_SetNRun_'+str(self.config['n_run']))
        time.sleep(delay)
        pbar.update(1)

        self.BCIServer.broadcastCmd('Cmd_MIOnline_SetNTrial_'+str(self.config['n_trial']))
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

        time.sleep(delay)
        pbar.close()

    def init_config(self, config):
        default_config = {
            'n_session': -1,
            'n_run': -1,
            'n_trial': -1,
            'DataPeriod': -1,
            'LeftHand': False,
            'RightHand': False,
            'Rest': False,
            'LeftFoot': False,
            'RightFoot': False,
            'Tongue': False
        }
        if config is None:
            self.config = default_config
        else:
            for k in default_config:
                self.config[k] = config[k]

    def run(self):
        self.reset()
        self.SendConfig()
        self.startListening()
        self.BCIServer.broadcastCmd('Cmd_MIOnline_Start')

    def reset(self):
        self.running_param['i_session'] = 0
        self.running_param['i_run'] = 0
        self.running_param['i_trial'] = 0

    def EventHandler(self, type):
        if type=='session start':
            self.running_param['i_session'] += 1
            self.running_param['i_run'] = -1
            self.running_param['i_trial'] = -1
            print('Session ', self.running_param['i_session']+1, ' Start')

        if type=='session end':
            print('Session ', self.running_param['i_session']+1, ' End')

        if type=='run start':
            self.running_param['i_run'] += 1
            self.running_param['i_trial'] = -1
            print('Run ', self.running_param['i_run']+1, ' Start')

        if type=='run end':
            print('Run ', self.running_param['i_run']+1, ' End')

        if type=='trial start':
            self.running_param['i_trial'] += 1
            print('Trial ', self.running_param['i_trial']+1, ' Start')

        if type=='trial end':
            print('Trial ', self.running_param['i_trial']+1, ' End')

        if type=='request data':
            print('MI Online Logic: Request received, generating Cmd...')

            stream_id = self.running_param['stream_id']
            dataPeriod = self.config['DataPeriod']
            client_id = self.running_param['client_id']
            data = self.BCIServer.streamClients[stream_id].Buffer.getData(dataPeriod=dataPeriod)
            data = np.expand_dims(data,axis=0)
            if self.preprocess is not None:
                data = self.preprocess.preprocess(data)
            y = self.trainned_model.predict(data)[0]

            if self.inverse_class_map is None:
                state = int(y)
            else:
                state = self.inverse_class_map[int(y)]
            self.BCIServer.valueService.SetValue('MIstate', state)
            self.BCIServer.valueService.UpdateValue(name='MIstate', value=state, conn=self.BCIServer.appClients[client_id])

    def startListening(self):
        self.BCIServer.eventService.typeChangedHandler.update({'MIOnline': self.EventHandler})
        self.running_param['is_Listening'] = True

    def stopListening(self):
        if self.running_param['is_Listening']:
            self.BCIServer.eventService.typeChangedHandler.pop('MIOnline')
            self.running_param['is_Listening'] = False

