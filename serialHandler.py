# Date 3/18/18
# Author: Aaron Rito
# Client: SARL Eggdicator
# this class handles the initialization and updating of the GUI, and the communication with the redstick.
# the comms are highly portable, the GUI elements should probably be in a different class, another time.

from PyQt5 import QtCore, QtWidgets, QtGui
import serial
#from BoxHandlerCore import BoxHandler
from mainFile import NewWindow
from settings_core import EggdicatorSettings
import JordanFile


class SerialThread(QtCore.QThread):
    updating_settings_signal = QtCore.pyqtSignal()
    updating_settings_signal_close = QtCore.pyqtSignal()

    def __init__(self, main_window, box_handler, comport_string):
        super(SerialThread, self).__init__()

        self.main_window = main_window
        self.box_handler = box_handler
        self.comport_string = comport_string
        self.main_ref = NewWindow
        self.in_buffer = ""
        self.arduino = serial.Serial(comport_string, 57600, bytesize=8, stopbits=1, timeout=None)
        self.should_run = True
        self.current_state = 0
        self.current_state_label = "Waiting for Command"
        self.current_state_strings = ["System Ready", "Waiting for Command", "Running Sort", "Running Maintenance"]
        self.update_flag = None
        self.settings_flag = False
        self.__connect_signals_to_slots()
        self.settings = QtCore.QSettings()
        self.settings_core = EggdicatorSettings(main_window)
        self.eggdicator = JordanFile.EggAI("./egg_clf.p", "/dev/ttyUSB0", train=False)

        self.start()

        #find layouts
        self.starts_button = self.main_window.start_button
        self.starts_button.clicked.connect(self.start_button_slot)
        self.pauses_button = self.main_window.pause_button
        self.pauses_button.clicked.connect(self.pause_button_slot)
        self.flushes_button = self.main_window.flush_button
        self.flushes_button.clicked.connect(self.flush_button_slot)
        self.aborts_button = self.main_window.abort_button
        self.aborts_button.clicked.connect(self.abort_button_slot)
        self.pump_1_power_box_x = self.main_window.pump_1_power_box
        self.pump_1_power_box_x.valueChanged.connect(self.pump_1_power_box_slot)
        self.pump_1_power_box_x.setValue(int(self.settings.value("pump_1_power")))
        self.pump_1_idle_box_x = self.main_window.pump_1_idle_box
        self.pump_1_idle_box_x.valueChanged.connect(self.pump_1_idle_box_slot)
        self.pump_1_idle_box_x.setValue(int(self.settings.value("pump_1_idle")))
        self.viewing_delay_box_x = self.main_window.view_delay_box
        self.viewing_delay_box_x.valueChanged.connect(self.view_delay_box_slot)
        self.viewing_delay_box_x.setValue(int(self.settings.value("viewing_clamp_delay")))
        self.y_delay_box_x = self.main_window.y_delay_box
        self.y_delay_box_x.valueChanged.connect(self.y_delay_box_slot)
        self.y_delay_box_x.setValue(int(self.settings.value("threeway_clamp_delay")))
        self.flush_dur_box_x = self.main_window.flush_dur_box
        self.flush_dur_box_x.valueChanged.connect(self.fluid_dur_box_slot)
        self.flush_dur_box_x.setValue(int(self.settings.value("fluid_duration")))
        self.pump_2_power_box_x = self.main_window.pump_2_power_box
        self.pump_2_power_box_x.valueChanged.connect(self.pump_2_power_box_slot)
        self.pump_2_power_box_x.setValue(int(self.settings.value("pump_2_power")))
        self.pump_2_idle_box_x = self.main_window.pump_2_idle_box
        self.pump_2_idle_box_x.valueChanged.connect(self.pump_2_idle_box_slot)
        self.pump_2_idle_box_x.setValue(int(self.settings.value("pump_2_idle")))
        self.update_buttons = self.main_window.update_button
        self.update_buttons.clicked.connect(self.update_button_slot)
        self.status_label = self.main_window.label_7
        self.status_label.setText(self.current_state_strings[0])

    def __connect_signals_to_slots(self):
        self.updating_settings_signal.connect(self.show_update_wait)
        self.main_window.stop_all_threads.connect(self.on_stop_all_threads_slot)

    def run(self):
        while self.should_run:
                if self.arduino.inWaiting():
                    in_byte = self.arduino.read().decode("utf-8")
                    self.in_buffer += in_byte
                    if in_byte == "\n":

                        if "u: " in self.in_buffer:
                            # The updates to a Shuttlebox are complete
                            self.update_flag = True
                            self.updating_settings_signal_close.emit()

                        if "a: " in self.in_buffer:
                            self.status_label.setText(self.current_state_strings[0])

                        if "x: " in self.in_buffer:
                            # The Shuttlebox is requesting it's settings
                            print("sending configs to box ")
                            self.send_to_box(self.settings_core.send_box_configs())
                        if "b: " in self.in_buffer:
                            # good egg done refresh motion cameras
                            self.eggdicator.signal_sent = False
                        if "c: " in self.in_buffer:
                            # bad egg done refresh motion cameras
                            self.eggdicator.signal_sent = False

                        print(self.in_buffer)
                        self.in_buffer = ""
                    self.msleep(50)
                    self.arduino.flush()

    def send_to_box(self, message):
        self.arduino.write(bytes(str(message)))

    def update_settings_slot(self):
        # Make sure the box isn't running before updating
        if self.current_state != 0:
            m = QtWidgets.QMessageBox()
            m.setInformativeText("Error: Cannot update Eggdicator while trial is running.")
            m.exec_()
        else:
            self.box_tab_widget.hide()
            m = QtWidgets.QMessageBox()
            m.setInformativeText("Are you sure you want to update Eggdicator?" + str(self.box_id))
            m.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            m.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
            m.setModal(True)
            x = m.exec_()

            if x == QtWidgets.QMessageBox.Ok:
                # Setup the waiting progress window
                self.welcome = QtWidgets.QProgressDialog()
                self.welcome.setCancelButton(None)
                self.welcome.setRange(0, 0)
                self.my_font = QtGui.QFont()
                self.my_font.setPointSizeF(11)
                self.my_font.setBold(True)
                self.welcome.setFont(self.my_font)
                self.welcome.setWindowTitle("SARL Eggdicator")
                self.welcome_label = QtWidgets.QLabel(
                    "       Please wait while The Eggdicator is updated.\n                  This may take a moment....")
                self.welcome_button = QtWidgets.QPushButton(None)
                self.welcome_window = QtWidgets.QGraphicsScene()
                self.welcome_image = QtGui.QImage("logo.png")
                self.lab = QtWidgets.QLabel()
                self.imageLabel = QtWidgets.QLabel()
                self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.welcome_image))
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(self.imageLabel)
                layout.addWidget(self.welcome_label)
                layout.addSpacing(50)
                self.welcome.setLayout(layout)
                self.welcome.resize(250, 200)
                self.welcome.setModal(True)
                self.welcome.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
                self.updating_settings_signal_close.connect(self.close_update)

                # Send the signals to update the box
                self.update_flag = False
                self.settings_core.update_settings()
                self.send_to_box(self.box_id)
                self.send_to_box(",")
                self.send_to_box("249")

                # Show the progress bar, it's closed by a signal from the serial input on line ..
                self.updating_settings_signal.emit()
                msg_box = QtWidgets.QMessageBox()
                msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg_box.setText("Settings Updated")
                msg_box.exec_()
                self.update_flag = False
                self.settings_flag = False

    # These two functions handle the progress window
    def show_update_wait(self):
        self.welcome.exec_()

    def close_update(self):
        self.welcome.done(0)

    def start_button_slot(self):
        self.status_label.setText(self.current_state_strings[2])
        # run Jordans thread here
        self.eggdicator.poll_cams(self)

    def pause_button_slot(self):
        self.status_label.setText(self.current_state_strings[1])
        # close Jordans thread
        self.eggdicator.kill_flag = True

    def flush_button_slot(self):
        self.send_to_box("u")
        self.status_label.setText(self.current_state_strings[3])

    def abort_button_slot(self):
        self.send_to_box("x")

    def update_button_slot(self):
        self.settings_core.update_settings()
        self.send_to_box("t")

    def pump_1_power_box_slot(self):
        print(self.pump_1_power_box_x.value())
        self.settings.setValue("pump_1_power", self.pump_1_power_box_x.value())

    def pump_1_idle_box_slot(self):
        print(self.pump_1_idle_box_x.value())
        self.settings.setValue("pump_1_idle", self.pump_1_idle_box_x.value())

    def view_delay_box_slot(self):
        print(self.viewing_delay_box_x.value())
        self.settings.setValue("viewing_clamp_delay", self.viewing_delay_box_x.value())

    def y_delay_box_slot(self):
        print(self.y_delay_box_x.value())
        self.settings.setValue("y_clamp_delay", self.y_delay_box_x.value())

    def fluid_dur_box_slot(self):
        self.settings.setValue("fluid_duration", self.flush_dur_box_x.value())

    def pump_2_power_box_slot(self):
        print(self.pump_2_power_box_x.value())
        self.settings.setValue("pump_2_power", self.pump_2_power_box_x.value())

    def pump_2_idle_box_slot(self):
        print(self.pump_2_idle_box_x.value())
        self.settings.setValue("pump_2_idle", self.pump_2_idle_box_x.value())

    def on_stop_all_threads_slot(self):
        self.send_to_box("x")
        self.should_run = False

