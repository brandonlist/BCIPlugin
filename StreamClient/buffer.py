import threading
import numpy as np

class EEGBuffer():
    def __init__(self, period, sfreq, n_chan):
        self.period = period
        self.sfreq = sfreq
        self.n_chan = n_chan

        self.bufferLength = int(period*sfreq)
        self.mutex = threading.Lock()

        self.Buffer = np.zeros((n_chan, self.bufferLength))
        self.putPos = 0

    def getData(self, dataPeriod):
        self.mutex.acquire()
        assert dataPeriod<self.period, 'can not request data that long'
        dataLength = int(self.sfreq*dataPeriod)
        startPos = self.putPos - dataLength
        if startPos>=0:
            data = self.Buffer[:,startPos:self.putPos]
        else:
            startPos = startPos % self.bufferLength
            if self.putPos==0:
                data = self.Buffer[:,startPos:]
            else:
                data = np.concatenate(
                    [self.Buffer[:,startPos:],self.Buffer[:,:self.putPos]],
                    axis=1
                )

        self.mutex.release()
        return data

    def putData(self, dataList):
        self.mutex.acquire()

        curPos = self.putPos
        endPos = int(self.putPos + dataList['receivedSamples'])
        if endPos<self.bufferLength:
            self.Buffer[:,curPos:endPos] = dataList['eeg']
            self.putPos = endPos
        else:
            preHalf = int(self.bufferLength-self.putPos)
            endPos = endPos % self.bufferLength
            self.Buffer[:,curPos:] = dataList['eeg'][:,0:preHalf]
            self.Buffer[:,:endPos] = dataList['eeg'][:,preHalf:]
            self.putPos = endPos

        self.mutex.release()


class RawBuffer():
    def __init__(self, raw):
        self.raw = raw
        self.sfreq = raw.info['sfreq']

        self.curSample = 0

    def getData(self, dataPeriod):
        start = self.curSample - int(self.sfreq*dataPeriod)
        data = self.raw.get_data(start=start,stop=self.curSample)
        return data

    def putData(self, dataList):
        print('Dont have to write this on simulator')
