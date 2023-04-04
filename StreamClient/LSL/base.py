import time

import mne
from pylsl import StreamInlet,resolve_stream
import numpy as np
from threading import Thread,Timer
import matplotlib.pyplot as plt

from StreamClient.buffer import EEGBuffer
from StreamClient.vis import StreamWindow
from StreamClient.record import StreamRecorder



def getArrayFromContainer(container):
    data = [c['eeg'] for c in container]
    eegArray = np.concatenate(data,axis=1)
    return eegArray

class LSLClient(StreamWindow, StreamRecorder):
    def __init__(self, ch_names, BufferPeriod=5, verbose=False, LSLType='EEG'):
        self.BufferPeriod = BufferPeriod
        self.verbose = verbose
        self.inlet = None
        self.LSLType = LSLType
        self.ch_names = ch_names

        self.isRecording = False
        self.isConnected = False

        self.Buffer = None
        self.buffer = None
        self.infoList = {
            'ch_names':[],
            'mne_info':None,
            'sfreq':None
        }

        self.raw_container = []
        self.curRawRecording = None
        self.raw_filename = None


    def startStreaming(self):
        if not self.isConnected:
            self.connectToLSL()
        self.initBuffer()

        self.receiving_loop()

    def connectToLSL(self):
        streams = resolve_stream('type', self.LSLType)
        inlet = StreamInlet(streams[0])
        sfreq = inlet.info().nominal_srate()
        self.infoList['sfreq'] = sfreq

        self.setInfoList(ch_names=self.ch_names,
                         ch_types=['eeg'] * len(self.ch_names))
        self.inlet = inlet

    def setInfoList(self, ch_names, ch_types):
        self.infoList['mne_info'] = mne.create_info(ch_names=ch_names,sfreq=self.infoList['sfreq'],ch_types='eeg')
        self.infoList['mne_info'].set_montage('standard_1020')
        self.infoList['ch_names'] = ch_names
        self.infoList['ch_types'] = ch_types

    def getInfoList(self):
        #自动获取InfoList
        pass

    def parseInfoList(self):
        return (self.infoList['ch_names'], self.infoList['sfreq'], self.infoList['ch_types'])

    def initBuffer(self):
        sfreq = self.inlet.info().nominal_srate()
        n_channel = self.inlet.info().channel_count()
        self.Buffer = EEGBuffer(period=self.BufferPeriod,
                                sfreq=sfreq,
                                n_chan=n_channel)

    def index2channel(self, idx):
        return self.infoList['ch_names'][idx]

    def index2pos(self, idx, dim=2):
        info = self.infoList['mne_info']
        if dim==2:
            return np.array([info['chs'][idx]['loc'][0],info['chs'][idx]['loc'][1]])
        if dim==3:
            return np.array([info['chs'][idx]['loc'][0],info['chs'][idx]['loc'][1],info['chs'][idx]['loc'][2]])

    def receiving_loop(self):
        Timer(interval=1, function=self.receiving_loop).start()

        dataList = {}
        samples, timestamps = self.inlet.pull_chunk(max_samples=int(self.infoList['sfreq']))
        if len(samples)>0:
            samples = np.transpose(np.array(samples))
            self.samples = samples
            dataList['eeg'] = samples
            dataList['receivedSamples'] = samples.shape[1]
            self.Buffer.putData(dataList)
            if self.isRecording:
                self.raw_container.append(dataList)


    def initTopoDisplay(self, channels):
        self.visTopoFigure = plt.figure()

        self.topoChannels = channels
        self.topoAxes = self.visTopoFigure.add_subplot(1, 1, 1)

    def initWaveDisplay(self, channels):
        self.visWaveFigure = plt.figure()

        self.waveChannels = channels
        self.waveAxes = []
        for i, chan in enumerate(self.waveChannels):
            self.waveAxes.append(self.visWaveFigure.add_subplot(int(str(len(channels)) + '1' + str(i + 1))))


    def startWaveDisplay(self):
        Timer(interval=1, function=self.startWaveDisplay).start()

        for i, chan in enumerate(self.waveChannels):
            self.waveAxes[i].cla()
            data = self.Buffer.getData(3)
            self.waveAxes[i].set_title(self.index2channel(i), loc='center')
            self.waveAxes[i].set_xticklabels('')
            self.waveAxes[i].plot(data[i])
            self.waveAxes[i].figure.canvas.draw()


    def startTopoDisplay(self):
        Timer(interval=3, function=self.startTopoDisplay).start()

        data = self.Buffer.getData(3)
        v = np.zeros(len(self.topoChannels))
        pos = np.zeros((len(self.topoChannels), 2))
        for i, chan in enumerate(self.topoChannels):
            v[i] = np.mean(data[chan])
            pos[i] = self.index2pos(chan)

        self.topoAxes.cla()
        mne.viz.plot_topomap(data=v, pos=pos, axes=self.topoAxes,
                             show_names=True, names=[self.index2channel(c) for c in self.topoChannels])

        self.topoAxes.figure.canvas.draw()


    def initRecording(self, filename):
        self.raw_filename = filename
        self.raw_container = []
        self.isRecording = False


    def startRecording(self, **kwargs):
        self.isRecording = True

    def stopRecording(self, **kwargs):
        self.isRecording = False
        ch_names, sfreq, ch_types = self.parseInfoList()
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage('standard_1020')
        data = getArrayFromContainer(self.raw_container)
        # convert to uv
        if np.abs(data.mean()) > 1e-3:
            data *= 1e-6
        self.curRawRecording = mne.io.RawArray(data=data, info=info)

ch_names = ['T7', 'T8', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4']
lc = LSLClient(ch_names=ch_names)
lc.startStreaming()
lc.initWaveDisplay(channels=[0,1,2])
lc.startWaveDisplay()
