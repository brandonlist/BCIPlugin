from abc import ABCMeta,abstractmethod

class StreamRecorder(metaclass=ABCMeta):
    @abstractmethod
    def startRecording(self, **kwargs):
        pass

    @abstractmethod
    def initRecording(self, **kwargs):
        pass

    @abstractmethod
    def stopRecording(self, **kwargs):
        pass



