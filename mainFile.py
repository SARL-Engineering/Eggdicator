# Date 3/18/18
# Author: Aaron Rito
# Client: SARL Eggdicator
# main thread for the Eggdicator. Nothing fancy yet.
import sys
import signal
from PyQt5 import QtCore, QtWidgets, QtGui, uic
import BoxHandlerCore
from settings_core import EggdicatorSettings

UI_FILE_PATH = "Eggdicator.ui"


class NewWindow(QtWidgets.QMainWindow):
    stop_all_threads = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(NewWindow, self).__init__(parent)
        uic.loadUi(UI_FILE_PATH, self)

        self.should_run = True
        self.run()

    def run(self):
        if self.should_run:
            self.settings_class = EggdicatorSettings(self)
            self.box_handler_class = BoxHandlerCore.BoxHandler(self)
            self.should_run = False


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QtWidgets.QApplication(sys.argv)

    myWindow = NewWindow()

    myWindow.show()

    app.exec_()
