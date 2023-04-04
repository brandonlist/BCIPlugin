import time

import numpy as np
from threading import Thread,Timer
from copy import deepcopy

from Paradigm.base import AsynParadigm
from tqdm import tqdm

class AsynMIParadigm(AsynParadigm):
    def __init__(self, BCIServer, preprocess, trainned_model, inverse_class_map=None,
                  log_func=print, dataPeriod=4):
        AsynParadigm.__init__(self, BCIServer=BCIServer)
        self.running_param = {
            'DataPeriod':dataPeriod,
            'stream_id':-1,
            'client_id':-1,
        }
        self.log_func = log_func
        self.ConfigWindow = None

        self.preprocess = preprocess
        self.trainned_model = trainned_model
        self.inverse_class_map = inverse_class_map

    def run(self, stream_id, client_id):
        self.running_param['stream_id'] = stream_id
        self.running_param['client_id'] = client_id

        self.running_thread = Thread(target=self.run_loop)
        self.running_thread.start()

    def run_loop(self):
        while True:
            time.sleep(5.)

            stream_id = self.running_param['stream_id']
            dataPeriod = self.running_param['DataPeriod']
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



