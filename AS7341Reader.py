import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib.util
from copy import deepcopy
from datetime import datetime
import sys

def import_module(path):
    spec = importlib.util.spec_from_file_location("cal_preprocessing.PreprocessingSOC", path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["cal_preprocessing.PreprocessingSOC"] = foo
    spec.loader.exec_module(foo)
    return foo

class AS7341Reader:
    def __init__(self, path, path_ref=None, path_dark=None, calibration_folder=None, path_length=None, offset=None):

        self.wl = [415, 445, 480, 515, 555, 590, 630, 680]
        self.calibration_folder = calibration_folder
        self.path = path

        if isinstance(path, list):
            self.data = []
            for p in path:
                data = pd.read_csv(p, index_col=0).drop(['F9 - 910/DarkRed'], axis=1)
                self.data.append(data)
            self.data = pd.concat(self.data)
        else:
            self.data = pd.read_csv(path + "/data.csv", index_col=0)
            self.data = self.data.drop(['F9 - 910/DarkRed'], axis=1)
        # Removing the NIR/ 910nm channel, it's just not clear if this is useful and accurate

        self.dates = []
        self.list_timestamp = []
        for t in self.data.index:  # Extract dates from the timestamp list
            self.list_timestamp.append(t)
            self.dates.append(datetime.fromtimestamp(t))

        #  Getting the dark and the ref
        if path_dark:
            self.dark = pd.read_csv(path_dark, index_col=0).drop(['F9 - 910/DarkRed'], axis=1).to_numpy()
        else:
            self.dark = (pd.read_csv(path + "/dark.csv", index_col=0))
            self.dark = self.dark.drop(['F9 - 910/DarkRed'], axis=1).to_numpy()
        if path_ref:
            self.ref = pd.read_csv(path_ref, index_col=0).drop(['F9 - 910/DarkRed'], axis=1).to_numpy()
        else:
            self.ref = pd.read_csv(path + "/ref.csv", index_col=0)
            self.ref = self.ref.drop(['F9 - 910/DarkRed'], axis=1).to_numpy()

        #  Calculating the absorbance
        self.abs = np.log10((self.ref - self.dark)/(self.data-self.dark)).to_numpy()
        self.abs = list(self.abs)
        abs_list = []
        for spec in self.abs:
            if offset:
                spec = spec + offset
            if path_length:
                spec = spec / path_length
            abs_list.append(pd.DataFrame({"Absorbance": spec}, index=self.wl))
        self.abs = abs_list



    def calculate_soc(self, index_cal=0, skip=1, idxmin=None, idxmax=None, arg=None):
        if isinstance(self.calibration_folder, list):
            preprocess = import_module(self.calibration_folder[index_cal] + "cal_preprocessing.py")
        else:
            preprocess = import_module(self.calibration_folder + "cal_preprocessing.py")

        if idxmin is None:
            idxmin = 0
        if idxmax is None:
            idxmax = len(self.abs)

        if isinstance(self.calibration_folder, list):
            self.preprocessing = preprocess.PreprocessingSOC(deepcopy(self.abs[idxmin:idxmax]),
                                                             self.calibration_folder[index_cal], skip=skip, arg=arg)
        else:
            self.preprocessing = preprocess.PreprocessingSOC(deepcopy(self.abs[idxmin:idxmax]),
                                                             self.calibration_folder, skip=skip, arg=arg)

        self.time_soc = []
        self.date_soc = []
        self.idx_soc = []
        for i, t in enumerate(self.list_timestamp[idxmin:idxmax]):
            if divmod(i, skip)[1] == 0:
                self.time_soc.append(t)
                self.date_soc.append(self.dates[i])
                self.idx_soc.append(i + idxmin)
        self.data_soc = pd.DataFrame({"Timestamp": self.time_soc, "Date": self.date_soc,
                                      "SOC": self.preprocessing.x}, index=self.idx_soc)
        self.data_c = pd.DataFrame({"Timestamp": self.time_soc, "Date": self.date_soc,
                                    "C": self.preprocessing.C}, index=self.idx_soc)



    def plot_a_bunch(self, skip=1, idx_min=None, idx_max=None):
        for i, spectrum in enumerate(self.abs[idx_min:idx_max]):
            if divmod(i, skip)[1] == 0:  # Plot every x spectra
                plt.plot(self.wl, spectrum, label=i)
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorbance")
        plt.show()

    def calculate_wl(self, wl):
        self.values = []
        idx = self.wl.index(wl)
        for spec in self.abs:
            self.values.append(spec.iloc[idx].values)
        self.intensity = pd.DataFrame({f"Absorbance ({wl}nm)": self.values}, index=self.list_timestamp)

    def calculate_plot_absorbance(self, wl):
        self.calculate_wl(wl)
        plt.plot(self.intensity.values)
        plt.ylabel(self.intensity.columns[0])
        plt.show()



if __name__ == '__main__':

    path = "/home/ange/MEGA/Postdoc_UC3M/Project_AS7341/Charge_1_pos/"
    r = AS7341Reader(path)
    r.calculate_plot_absorbance(680)
