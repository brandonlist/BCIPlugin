from abc import ABCMeta, abstractmethod

class EEGDecoder(metaclass=ABCMeta):
    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def score(self, **kwargs):
        pass

    @abstractmethod
    def train_finetune(self, **kwargs):
        pass


class Inspector(metaclass=ABCMeta):
    @abstractmethod
    def inspect(self, **kwargs):
        pass





