from BCIServer.base import BCIServer

from Paradigm.MI.MICalibrate import MICalibrateParadigm
from StreamClient.LSL.base import LSLClient

file_dir = '.\\Tutorials\\data\\SynMI_Experiment\\'
save_name = 'Dateset_MICalibrate_OpenBCI'

server = BCIServer()
server.run()

ch_names = ['T7', 'T8', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4']
sc = LSLClient(ch_names=ch_names)
sc.startStreaming()
server.loadStreamClient(sc)
i_streamClient = list(server.streamClients.keys())[0]

server.listenUnityAppClient()


server.loadParadigm(MICalibrateParadigm(BCIServer=server))
i_paradigm = list(server.paradigms.keys())[0]

server.paradigms[i_paradigm].running_param['stream_id'] = i_streamClient
server.paradigms[i_paradigm].config['n_session'] = 1
server.paradigms[i_paradigm].config['n_run'] = 1
server.paradigms[i_paradigm].config['n_trial'] = 10
server.paradigms[i_paradigm].config['DataPeriod'] = 4
server.paradigms[i_paradigm].config['TrialLength'] = 5
server.paradigms[i_paradigm].config['LeftHand'] = True
server.paradigms[i_paradigm].config['RightFoot'] = True

server.paradigms[i_paradigm].run()


server.paradigms[i_paradigm].stop()
info = server.streamClients[i_streamClient].infoList['mne_info']
windowedDataset, class_map = server.paradigms[i_paradigm].createDataset(trial_start=0,
                                                                 trial_length=4,
                                                                 subject_id=0,
                                                                 info=info)
server.paradigms[i_paradigm].save_calibrate_to_pkl_file(windowedDataset=windowedDataset,
                                                      class_map=class_map,
                                                        filepath=file_dir,
                                                        filename=save_name)


