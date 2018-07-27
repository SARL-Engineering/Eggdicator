# Date 3/18/18
# Author: Aaron Rito
# Client: SARL Eggdicator
# This is a typical Qsettings class, there is a method to update the settings during runtime.
from PyQt5 import QtCore


class EggdicatorSettings(QtCore.QObject):

    def __init__(self, main_window):
        QtCore.QObject.__init__(self)

        # Reference to highest level window
        self.main_window = main_window
        self.setup_settings()
        self.settings = QtCore.QSettings()

        self.boxes_configs_array = []

        # run
        self.settings.setFallbacksEnabled(False)

        # If things get messed up in the registry, uncomment this line and recomplie. Make sure to re-comment.
        #self.settings.clear()
        self.load_settings()

    @staticmethod
    def setup_settings():
        QtCore.QCoreApplication.setOrganizationName("OSU SARL")
        QtCore.QCoreApplication.setOrganizationDomain("ehsc.oregonstate.edu/sarl")
        QtCore.QCoreApplication.setApplicationName("Eggdicator")

    def load_settings(self):
        self.settings.setValue("pump_1_power", self.settings.value("pump_1_power", 239))
        self.settings.setValue("pump_1_idle", self.settings.value("pump_1_idle", 15))
        self.settings.setValue("viewing_clamp_delay", self.settings.value("viewing_clamp_delay", 500))
        self.settings.setValue("threeway_clamp_delay", self.settings.value("threeway_clamp_delay", 400))
        self.settings.setValue("fluid_duration", self.settings.value("fluid_duration", 30000))
        self.settings.setValue("pump_2_power", self.settings.value("pump_2_power", 239))
        self.settings.setValue("pump_2_idle", self.settings.value("pump_2_idle", 1))

        self.boxes_configs_array = [self.settings.value("pump_1_power", 250), self.settings.value("pump_1_idle", 12),
                                    self.settings.value("viewing_clamp_delay", 1000), self.settings.value(
                                    "threeway_clamp_delay", 2000), self.settings.value("fluid_duration", 30000),
                                    self.settings.value("pump_2_power", 1), self.settings.value("pump_2_idle", 1)]

    def send_box_configs(self):
        print(self.boxes_configs_array)
        return self.boxes_configs_array

    def update_settings(self):
        self.boxes_configs_array = [self.settings.value("pump_1_power"), self.settings.value("pump_1_idle"),
                                    self.settings.value("viewing_clamp_delay"), self.settings.value(
                                    "threeway_clamp_delay"), self.settings.value("fluid_duration"),
                                    self.settings.value("pump_2_power"), self.settings.value("pump_2_idle")]



