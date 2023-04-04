import socket
from threading import Thread

from BCIServer.Services.ValueService import ValueService
from BCIServer.Services.EventService import EventService
from bGUI.mainWindow import BCIplugWindow
from PluginCore.base import PluginCore

from PyQt5 import uic
from PyQt5.Qt import QApplication,QWidget,QThread
from PyQt5.QtWidgets import QListView,QAbstractItemView
from PyQt5.QtCore import  QStringListModel
import sys


class BCIServer():
    def __init__(self, host='', port=50000, log_func=None, value_func=None, pluginCore=None):
        self.host = host
        self.port = port

        self.streamClients = {}
        self.n_streamClient = 0
        self.i_streamClient = -1

        self.appClients = {}
        self.n_appClient = 0
        self.i_appClient = -1

        self.appMsgHandlers = [self.cmdMessageHandler]

        self.eventService = EventService()
        self.appMsgHandlers.append(self.eventService.MessageHandler)
        self.valueService = ValueService()
        self.appMsgHandlers.append(self.valueService.MessageHandler)

        self.appSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.paradigms = {}
        self.i_paradigm = -1
        self.n_paradigm = 0

        self.results = {}
        self.n_result = 0
        self.i_result = -1

        self.values = {}

        self.log_func = log_func
        self.value_func = value_func

        self.window = None

        if pluginCore is None:
            self.pluginCore = PluginCore(preprocess={},algorithms={},datasets={},modules={},inspectors={})
        else:
            self.pluginCore = pluginCore


    def run(self):
        try:
            self.appSocket.bind((self.host, self.port))
            self.appSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except socket.error as err:
            self.log("Error binding to server: {}".format(err))
        self.appSocket.listen(10)
        if not self.host:
            self.host = "localhost"

        self.log('Unity server is running at {}:{}'.format(self.host, self.port))

    def GUI(self):
        self.window = BCIplugWindow(BCIServer=self)
        self.log_func = self.window.log
        self.value_func = self.window.updateValue
        self.window.show()

        self.window = None

    def log(self, msg):
        print(msg)
        if self.log_func is not None:
            self.log_func(msg)

    def cmdMessageHandler(self, msg):
        msg_split = msg.split('_')
        if msg_split[0]=='cmd':
            self.log('BCIServer received Cmd from AppClient: '+ msg)
            for exp in self.paradigms.values():
                try:
                    exp.CmdHandler(msg_split=msg_split)
                except:
                    continue
        elif msg_split[0]=='Value':
            if self.value_func:
                self.value_func()

    def broadcastCmd(self, cmd):
        for conn in self.appClients.values():
            conn.sendall(str.encode(cmd))

    def listenUnityAppClient(self):
        self.log('Unity server listening for application client connection...')
        conn, address = self.appSocket.accept()

        thread = Thread(target=self.appClientConnectHandler, args=(conn, address))
        thread.daemon = True
        thread.start()

        self.n_appClient += 1
        default_name = 'UnityAppClient_' + str(self.n_appClient)
        self.appClients[default_name] = conn
        self.i_appClient = self.n_appClient - 1

    def loadStreamClient(self, client):
        self.log('BCI Server connecting to Streaming Client...')

        self.n_streamClient += 1
        default_name = 'StreamClient_' + str(self.n_streamClient)
        self.streamClients[default_name] = client
        self.i_streamClient = self.n_streamClient - 1

    def loadParadigm(self, paradigm):
        self.log('BCI Server loading paradigm...')

        self.n_paradigm += 1
        default_name = 'Paradigm_' + str(self.n_paradigm) + ': ' + str(paradigm.__class__.__name__)
        self.paradigms[default_name] = paradigm
        self.i_paradigm = self.n_paradigm - 1

    def loadResult(self, filepath):
        self.log('BCI Server loading result...')

        self.n_result += 1

        #TODO:运行时有且只有一个core，使用当前core加载reuslt,之后加入其他core的处理逻辑

        default_name = 'Result_' + str(self.n_result) + ':' + filepath.split('_')[0]
        df = PluginCore.read_df_from_file(filepath)
        self.results[default_name] = df
        self.i_result = self.n_result - 1

    def startStreamClient(self):
        cur_key = list(self.streamClients.keys())[self.i_streamClient]
        self.streamClients[cur_key].startStreaming()

    def stopStreamClient(self):
        cur_key = list(self.streamClients.keys())[self.i_streamClient]
        self.streamClients[cur_key].stopStreaming()


    def appClientConnectHandler(self, conn, a):
        self.log("App Client {}:{} connected...".format(a[0], a[1]))
        while True:
            try:
                data = conn.recv(1024)
                message = data.decode('UTF-8')
                for messageHandler in self.appMsgHandlers:
                    messageHandler(message)
            except socket.error as e:
                self.log("Error! {}".format(e))
                break
        conn.close()

    def shutdown(self):
        for connection in self.appClients.values():
                connection.close()
        self.appSocket.close()

        #TODO: Shut down Streaming Clients!
        self.log("Unity server shutdown successfully")






