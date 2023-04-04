import sys
import time

from PyQt5 import uic
from PyQt5.Qt import QApplication,QWidget,QThread,QFileDialog
from PyQt5.QtWidgets import QListView,QAbstractItemView
from PyQt5.QtCore import  QStringListModel

from StreamClient.Curry.base import CurryClient
from Paradigm.MI.MICalibrate import MICalibrateParadigm
from PluginCore.base import PluginCore

ui_filepath = r'.\bGUI\mainWindow.ui'

#TODO:用来做耗时的操作
class WindowThread(QThread):
    def __init__(self):
        super(WindowThread, self).__init__()

    def run(self):
        pass


class BCIplugWindow(QWidget):
    def __init__(self, BCIServer):
        super(BCIplugWindow, self).__init__()

        self.initUI()
        self.BCIServer = BCIServer
        self.loadServer()

    def log(self, msg):
        self.plainTextLog.insertPlainText(msg + '\n')

    def initUI(self):
        self.ui = uic.loadUi(ui_filepath)

        self.actionNCSCFrom_Curry8 = self.ui.actionNCSCFrom_Curry8
        self.actionNCSCFrom_Curry8.triggered.connect(self.addCurry8StreamClient)
        self.btnStartStreamClient = self.ui.btnStartStreamClient
        self.btnStartStreamClient.clicked.connect(self.startClientStreaming)
        self.listViewStreamClients = self.ui.listStreamClient

        self.actionFrom_Unity = self.ui.actionFrom_Unity
        self.actionFrom_Unity.triggered.connect(self.addUnityAppClient)
        self.listAppClient = self.ui.listAppClient

        self.actionMI_Calibrate = self.ui.actionMI_Calibrate
        self.actionMI_Calibrate.triggered.connect(self.addMICalibrateParadigm)
        self.listParadigm = self.ui.listParadigm

        self.btnConfigParadigm = self.ui.btnConfigParadigm
        self.btnConfigParadigm.clicked.connect(self.configParadigm)
        self.btnRunParadigm = self.ui.btnRunParadigm
        self.btnRunParadigm.clicked.connect(self.runParadigm)

        self.btnLoadResult = self.ui.btnLoadResult
        self.btnLoadResult.clicked.connect(self.loadResult)

        self.btnUpdateCore = self.ui.btnUpdateCore
        self.btnUpdateCore.clicked.connect(self.updateStrPluginCore)

        self.listPC_datasets = self.ui.listDatasets
        self.listPC_algorithms = self.ui.listAlgorithm
        self.listPC_modules = self.ui.listModule
        self.listPC_inspectors = self.ui.listInspector
        self.listPC_processers = self.ui.listProcessor


        self.plainTextLog = self.ui.plainTextLog

        self.listValue = self.ui.listValue

        self.listResult = self.ui.listResult

    def loadServer(self):
        self.updateStrStreamClients()
        self.updateStrAppClients()
        self.updateStrParadigms()
        self.updateValue()
        self.updateStrPluginCore()

    def show(self):
        self.ui.show()

    def updateStrPluginCore(self):
        pluginCore = self.BCIServer.pluginCore
        datasets = pluginCore.datasets
        algorithms = pluginCore.algorithms
        modules = pluginCore.modules
        inspectors = pluginCore.inspectors
        processors = pluginCore.preprocess

        self.listPC_datasets.clear()
        for k in datasets:
            self.listPC_datasets.addItem(str(k)+': '+str(datasets[k].__class__))

        self.listPC_algorithms.clear()
        for k in algorithms:
            self.listPC_algorithms.addItem(str(k)+': '+str(algorithms[k].__class__))

        self.listPC_modules.clear()
        for k in modules:
            self.listPC_modules.addItem(str(k)+': '+str(modules[k].__class__))

        self.listPC_inspectors.clear()
        for k in inspectors:
            self.listPC_inspectors.addItem(str(k)+': '+str(inspectors[k].__class__))

        self.listPC_processers.clear()
        for k in processors:
            self.listPC_processers.addItem(str(k)+': '+str(processors[k].__class__))





    def updateStrParadigms(self):
        cur_keys = list(self.BCIServer.paradigms.keys())
        self.listParadigm.clear()
        for k in cur_keys:
            self.listParadigm.addItem(k)

    def updateStrStreamClients(self):
        cur_keys = list(self.BCIServer.streamClients.keys())
        self.listViewStreamClients.clear()
        for k in cur_keys:
            self.listViewStreamClients.addItem(k)

    def updateStrAppClients(self):
        cur_keys = list(self.BCIServer.appClients.keys())
        self.listAppClient.clear()
        for k in cur_keys:
            self.listAppClient.addItem(k)

    def updateValue(self):
        self.listValue.clear()
        values = self.BCIServer.valueService.values
        for k in values:
            self.listValue.addItem(k+': '+str(values[k]))

    def loadResult(self):
        filepath = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.csv , *.xls)')
        self.BCIServer.loadResult(filepath[0])
        self.updateResult()

    def updateResult(self):
        cur_keys = list(self.BCIServer.results.keys())
        self.listResult.clear()
        for k in cur_keys:
            self.listResult.addItem(k)

    def startClientStreaming(self):
        cur_key = str(self.listViewStreamClients.currentItem().data(0))
        self.BCIServer.streamClients[cur_key].startStreaming()


    def addCurry8StreamClient(self):
        curryClient = CurryClient()
        assert (self.BCIServer is not None), print('Run BCI Server first..')
        self.BCIServer.loadStreamClient(curryClient)
        self.updateStrStreamClients()

    def addUnityAppClient(self):
        assert (self.BCIServer is not None), print('Run BCI Server first..')
        self.BCIServer.listenUnityAppClient()
        self.updateStrAppClients()

    def addMICalibrateParadigm(self):
        assert (self.BCIServer is not None), print('Run BCI Server first..')
        MIparadigm = MICalibrateParadigm(BCIServer=self.BCIServer,log_func=self.log)
        self.BCIServer.loadParadigm(MIparadigm)
        self.updateStrParadigms()

    def configParadigm(self):
        cur_key = str(self.listParadigm.currentItem().data(0))
        paradigm = self.BCIServer.paradigms[cur_key]
        paradigm.configParadigm()

    def runParadigm(self):
        cur_key = str(self.listParadigm.currentItem().data(0))
        paradigm = self.BCIServer.paradigms[cur_key]
        paradigm.run()
