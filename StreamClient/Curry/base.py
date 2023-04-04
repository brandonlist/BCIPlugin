import socket
from threading import Thread,Timer
import matplotlib.pyplot as plt
import mne
import numpy as np

from StreamClient.Curry.CurryProtocol import *
from StreamClient.buffer import EEGBuffer
from StreamClient.vis import StreamWindow
from StreamClient.record import StreamRecorder


def isContinues(container):
    flag = True
    startPoints = [c['startSample'] for c in container]
    received = [c['receivedSamples'] for c in container]
    for i in range(len(startPoints)-1):
        if int(startPoints[i]+received[i]) != int(startPoints[i+1]):
            flag = False
    return flag


def getArrayFromContainer(container):
    startPoints = [c['startSample'] for c in container]
    received = [c['receivedSamples'] for c in container]
    data = [c['eeg'] for c in container]
    assert len(startPoints)==len(received)
    if(isContinues(container)):
        eegArray = np.concatenate(data,axis=1)
    else:
        eegArray = None
        print('Data is not Continuous. returning None')
    return (startPoints[0], eegArray)


class CurryClient(StreamWindow, StreamRecorder):
    def __init__(self, host='localhost', port=4455, BufferPeriod=5,
                 verbose=False, recordOnStart=False):
        self.host = host
        self.tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = port
        self.isConnected = False

        self.buffer = None
        self.infoList = None
        self.basicInfo = None
        self.Buffer = None
        self.BufferPeriod = BufferPeriod
        self.verbose = verbose

        #For Recording implementation
        self.raw_filename = None
        self.raw_container = []
        self.isRecording = False
        self.recordOnStart = recordOnStart
        self.curRawRecording = None

    def startStreaming(self):
        if not self.isConnected:
            self.connectToCurry()
        self.initBuffer()
        self.infoList = self.getInfoList()

        if self.recordOnStart:
            self.isRecording = True

        streamThread = Thread(target=self.receiving_loop)
        streamThread.start()

    def connectToCurry(self):
        self.tcpClient.connect((self.host,self.port))
        self.isConnected = True

    def initBuffer(self):
        (_, basicInfo) = self.getBasicInfo()
        sfreq = basicInfo['sampleRate']
        n_chan = basicInfo['eegChan']
        self.Buffer = EEGBuffer(period=self.BufferPeriod,
                                sfreq=sfreq,
                                n_chan=n_chan)

    def getBasicInfo(self):
        if not self.isConnected:
            self.connectToCurry()

        [status, basicInfo] = clientGetBasicInfo(con=self.tcpClient, handles=None)
        return (status, basicInfo)

    def getInfoList(self):
        if not self.isConnected:
            self.connectToCurry()
        [status, basicInfo] = clientGetBasicInfo(con=self.tcpClient, handles=None)
        if status:
            self.basicInfo = basicInfo
            [status, infoList] = clientGetChannelInfoList(con=self.tcpClient, numChannels=basicInfo['eegChan'], handles=None)
        else:
            infoList = None
        return (status, infoList)

    def receiving_loop(self):
        timeout = 10
        count = 0

        [status, _ ] = clientCheckNetStreamingVersion(self.tcpClient)
        if status:
            [status, basicInfo] = clientGetBasicInfo(con=self.tcpClient, handles=None)
            if status:
                [status, infoList] = clientGetChannelInfoList(con=self.tcpClient,numChannels=basicInfo['eegChan'],handles=None)
                if status:
                    while count<timeout:
                        [status, dataList] = clientRequestDataPacket(con=self.tcpClient, basicInfo=basicInfo, infoList=infoList,
                                                                     handles=None, startStop=1, init=False, verbose=self.verbose)
                        if 'eeg' in dataList.keys():
                            self.Buffer.putData(dataList)
                            self.buffer = dataList['eeg']
                            if self.isRecording:
                                self.raw_container.append(dataList)
                        else:
                            continue
                        if status:
                            count = 0
                            init = True
                        else:
                            count += 1

    def index2channel(self, idx):
        return self.infoList[1][idx][0]['chanLabel']

    def index2pos(self, idx, dim=2):
        if dim==2:
            return np.array([self.infoList[1][idx][0]['posX'],self.infoList[1][idx][0]['posY']])
        if dim==3:
            return np.array([self.infoList[1][idx][0]['posX'], self.infoList[1][idx][0]['posY'], self.infoList[1][idx][0]['posZ']])

    def parseInfoList(self):
        ch_names = [i[0]['chanLabel'] for i in self.infoList[1]]
        sfreq = self.basicInfo['sampleRate']
        ch_types = [CurryChannelTypes[i[0]['chanType']] for i in self.infoList[1]]
        return (ch_names, sfreq, ch_types)

    def initWaveDisplay(self, channels):
        self.visWaveFigure = plt.figure()

        self.waveChannels = channels
        self.waveAxes = []
        for i,chan in enumerate(self.waveChannels):
            self.waveAxes.append(self.visWaveFigure.add_subplot(int(str(len(channels))+'1'+str(i+1))))

    def startWaveDisplay(self):
        Timer(interval=1,function=self.startWaveDisplay).start()

        for i,chan in enumerate(self.waveChannels):
            self.waveAxes[i].cla()
            data = self.Buffer.getData(3)
            self.waveAxes[i].set_title(self.index2channel(i),loc='center')
            self.waveAxes[i].set_xticklabels('')
            self.waveAxes[i].plot(data[i])
            self.waveAxes[i].figure.canvas.draw()

    def initTopoDisplay(self, channels):
        self.visTopoFigure = plt.figure()

        self.topoChannels = channels
        self.topoAxes = self.visTopoFigure.add_subplot(1,1,1)

    def startTopoDisplay(self):
        Timer(interval=5,function=self.startTopoDisplay).start()

        data = self.Buffer.getData(5)
        v = np.zeros(len(self.topoChannels))
        pos = np.zeros((len(self.topoChannels),2))
        for i,chan in enumerate(self.topoChannels):
            v[i] = np.mean(data[chan])
            pos[i] = self.index2pos(chan)

        self.topoAxes.cla()
        mne.viz.plot_topomap(data=v,pos=pos,axes=self.topoAxes,
                             show_names=True,names=[self.index2channel(c) for c in self.topoChannels])

        self.topoAxes.figure.canvas.draw()

    def initRecording(self, filename):
        self.raw_filename = filename
        self.raw_container = []
        self.isRecording = False

    def startRecording(self):
        self.isRecording = True

    def stopRecording(self):
        self.isRecording = False
        ch_names, sfreq, ch_types = self.parseInfoList()
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        startSample, data = getArrayFromContainer(self.raw_container)
        #convert to uv
        if data.mean()>1e-3:
            data *= 1e-6
        self.curRawRecording = mne.io.RawArray(data=data, info=info, first_samp=startSample)
