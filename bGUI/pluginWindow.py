from PyQt5 import uic
from PyQt5.Qt import QApplication,QWidget,QThread,QCheckBox
from PyQt5.QtWidgets import QListView,QAbstractItemView
from PyQt5.QtCore import  QStringListModel


MICalibrate_ui = r'.\bGUI\MICalibrate.ui'

class WinMICalibrate(QWidget):
    def __init__(self, paradigm):
        super(WinMICalibrate, self).__init__()
        self.MICalibrateParadigm = paradigm
        self.initUI()

    def initUI(self):
        self.ui = uic.loadUi(MICalibrate_ui)

        self.cBoxLeftFoot = self.ui.cBoxLeftFoot
        self.cBoxLeftFoot.setChecked(self.MICalibrateParadigm.config['LeftFoot'])
        self.cBoxLeftFoot.stateChanged.connect(lambda: self.OnMItypeChange(self.cBoxLeftFoot, "LeftFoot"))

        self.cBoxLeftHand = self.ui.cBoxLeftHand
        self.cBoxLeftHand.setChecked(self.MICalibrateParadigm.config['LeftHand'])
        self.cBoxLeftHand.stateChanged.connect(lambda: self.OnMItypeChange(self.cBoxLeftHand, "LeftHand"))

        self.cBoxRest = self.ui.cBoxRest
        self.cBoxRest.setChecked(self.MICalibrateParadigm.config['Rest'])
        self.cBoxRest.stateChanged.connect(lambda: self.OnMItypeChange(self.cBoxRest, "Rest"))

        self.cBoxRightHand = self.ui.cBoxRightHand
        self.cBoxRightHand.setChecked(self.MICalibrateParadigm.config['RightHand'])
        self.cBoxRightHand.stateChanged.connect(lambda: self.OnMItypeChange(self.cBoxRightHand, "RightHand"))

        self.cBoxRightFoot = self.ui.cBoxRightFoot
        self.cBoxRightFoot.setChecked(self.MICalibrateParadigm.config['RightFoot'])
        self.cBoxRightFoot.stateChanged.connect(lambda: self.OnMItypeChange(self.cBoxRightFoot, "RightFoot"))

        self.cBoxTongue = self.ui.cBoxTongue
        self.cBoxTongue.setChecked(self.MICalibrateParadigm.config['Tongue'])
        self.cBoxTongue.stateChanged.connect(lambda: self.OnMItypeChange(self.cBoxTongue, "RightFoot"))

        self.linen_session = self.ui.linen_session
        self.linen_session.setText(str(self.MICalibrateParadigm.config['n_session']))

        self.linen_run = self.ui.linen_run
        self.linen_run.setText(str(self.MICalibrateParadigm.config['n_run']))

        self.linen_trial = self.ui.linen_trial
        self.linen_trial.setText(str(self.MICalibrateParadigm.config['n_trial']))

        self.lineDataPeriod = self.ui.lineDataPeriod
        self.lineDataPeriod.setText(str(self.MICalibrateParadigm.config['DataPeriod']))

        self.lineTrialLength = self.ui.lineTrialLength
        self.lineTrialLength.setText(str(self.MICalibrateParadigm.config['TrialLength']))

        self.btnConfirm = self.ui.btnConfirm
        self.btnConfirm.clicked.connect(self.Confirm)

    def show(self):
        self.ui.show()

    def OnMItypeChange(self, btn, name):
        if btn.isChecked():
            self.MICalibrateParadigm.config[name] = True
        else:
            self.MICalibrateParadigm.config[name] = False

    def Confirm(self):
        self.MICalibrateParadigm.config['n_session'] = int(self.linen_session.text())
        self.MICalibrateParadigm.config['n_run'] = int(self.linen_run.text())
        self.MICalibrateParadigm.config['n_trial'] = int(self.linen_trial.text())
        self.MICalibrateParadigm.config['DataPeriod'] = int(self.lineDataPeriod.text())
        self.MICalibrateParadigm.config['TrialLength'] = int(self.lineTrialLength.text())





