# Date 3/18/18
# Author: Aaron Rito
# Client: SARL Eggdicator
# This class handles opening a thread for each arduino type device connected
# NOTE: This not needed for 1 device, but I have it here in case of expansion

from PyQt5 import QtCore, QtWidgets
import serial
from serial.tools.list_ports import comports
import serialHandler


class BoxHandler(QtCore.QThread):

    def __init__(self, main_window):
        super(BoxHandler, self).__init__()

        # GUI Object References
        self.main_window = main_window
        # this will hold the array of serial handler threads.
        self.thread_instances = []

        # more GUI stuff
        self.__connect_signals_to_slots()
        # variables
        self.box_count = 0
        self.should_run = True
        self.start()

    def __connect_signals_to_slots(self):

        # make an iterable an array to sort the COM ports
        my_it = sorted(comports())
        ports = []  # type: serial.Serial

        for n, (port, desc, hwid) in enumerate(my_it, 1):
            # print "    desc: {}\n".format(desc)
            # print "    hwid: {}\n".format(hwid)
            ports.append(port)
            # print ports
            # if pid == "2341":
            #    print "this worked"

        # !!!!IMPORTANT!!!! For every device found, open a new serial thread.
        for port in ports:
            if port == "COM1" or port == "COM3":
                continue
            self.thread_instances.append(serialHandler.SerialThread(self.main_window, self, port))

    def on_stop_all_threads_slot(self):
        print("Box handler exiting")
        self.should_run = False
        for thread in self.thread_instances:
            thread.wait()
