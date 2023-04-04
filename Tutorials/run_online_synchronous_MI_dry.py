import os
import pickle

from Paradigm.MI.MIOnline import MIOnlineParadigm
from PluginCore.base import PluginCore
from StreamClient.LSL.base import LSLClient
from BCIServer.base import BCIServer


file_dir = '.\\Tutorials\\data\\SynMI_Experiment'
core_dir = file_dir
core_path = os.path.join(core_dir, 'CoreModelTrain.pkl')
core = PluginCore(preprocess={},algorithms={},datasets={})
core.load_core_from_file(core_path)

with open(os.path.join(file_dir,'Dateset_MICalibrate_OpenBCI.pkl'),'rb') as f:
    calibrate_data = pickle.load(f)
inverse_class_map = calibrate_data['inverse_class_map']

server = BCIServer(pluginCore=core)
server.run()

server.listenUnityAppClient()
i_appClient = list(server.appClients.keys())[0]

ch_names = ['T7', 'T8', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4']
sc = LSLClient(ch_names=ch_names)
sc.startStreaming()

server.loadStreamClient(sc)

server.loadParadigm(MIOnlineParadigm(BCIServer=server,preprocess=None,inverse_class_map=inverse_class_map,
                                     trainned_model=core.modules['trained_module']))

i_paradigm = list(server.paradigms.keys())[0]
i_streamClient = list(server.streamClients.keys())[0]

server.paradigms[i_paradigm].running_param['stream_id'] = i_streamClient
server.paradigms[i_paradigm].running_param['client_id'] = i_appClient


server.paradigms[i_paradigm].config['n_session'] = 1
server.paradigms[i_paradigm].config['n_run'] = 1
server.paradigms[i_paradigm].config['n_trial'] = 10
server.paradigms[i_paradigm].config['DataPeriod'] = 4
server.paradigms[i_paradigm].config['LeftHand'] = True
server.paradigms[i_paradigm].config['RightFoot'] = True

server.paradigms[i_paradigm].run()

server.paradigms[i_paradigm].stop()

