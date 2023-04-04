"""
三种方式：
1. AsynParadigm，Update函数自动按定时器执行, 继承该类说明该Paradigm只会进行单一过程
2. SynParadigm，在event值发生变化时，会根据Event自动执行回调，继承该类说明该Paradigm只会进行单一过程
3. 两种Paradigm，在CmdDefine里定义cmd调用相对应的函数
"""



class SynParadigm():
    def __init__(self, BCIServer):
        self.BCIServer = BCIServer

    def EventHandler(self, **kwargs):
        pass

    def startListening(self, **kwargs):
        pass

    def stopListening(self, **kwargs):
        pass

    def configParadigm(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass


class AsynParadigm():
    def __init__(self, BCIServer):
        self.BCIServer = BCIServer

    def run(self, **kwargs):
        pass


#Need more consideration, 每个core都要能产生result，paradigm也要能产生result
class Result():
    def __init__(self, Id, name):
        self.Id = Id
        self.name = name
        self.datasets = {}





