import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def return_diff(x, spectrum, wl1, wl2, e4, e45, e5, k, p):
    spectrum = spectrum[wl1:wl2]
    e5 = e5[wl1:wl2]
    e4 = e4[wl1:wl2]
    e45 = e45[wl1:wl2]
    chi = k / (k * x[0] + 1)
    C45 = (1 - np.sqrt(1 - 4 * chi ** 2 * x[1] * (1 - x[1]) * x[0] ** 2)) / 2 / chi
    diff = np.sum(
        abs(spectrum - e5*(x[0]*x[1]-C45)**p - e4 * ((1-x[1])*x[0]-C45) - e45*C45), axis=0)
    return diff



class PreprocessingSOC:

    """"Algorithm that performs deconvolution of the spectrum using epsilon 4,
    epsilon 5 and epsilon 45 (the complex absorbance)"""

    def __init__(self, list_spectra, calibration_folder, skip=1, arg=None):

        self.wl = [415, 445, 480, 515, 555, 590, 630, 680]
        self.list_spectra = []
        for spec in list_spectra:
            self.list_spectra.append(spec["Absorbance"].values)
        self.e45 = pd.read_csv(calibration_folder + "/e45.csv", index_col=0)
        self.e45 = self.e45["Absorbance"].values
        self.e4 = pd.read_csv(calibration_folder + "/e4.csv", index_col=0)
        self.e4 = self.e4["Absorbance"].values
        self.e5 = pd.read_csv(calibration_folder + "/e5.csv", index_col=0)
        self.e5 = self.e5["Absorbance"].values
        self.p = np.load(calibration_folder + "p.npy")
        self.k = np.load(calibration_folder + "K.npy")
        self.list_fit = []
        self.result = []
        self.index = []
        self.err = []

        method = None
        method = "SLSQP"
        options = {"ftol": 1e-12, "maxfun": 1000}
        for i, spec in enumerate(self.list_spectra):
            if divmod(i, skip)[1] == 0:
                x = minimize(return_diff, [0.5, 0.5], args=(spec, 1, 6, self.e4, self.e45, self.e5, self.k, self.p),
                             bounds=((1, 2.2), (0, 1)), options=options, method=method)
                print(f"{i}/{len(self.list_spectra)}")
                print(x.success)
                print(x.x)
                self.result.append(x.x)
                self.list_fit.append(x)
                self.err.append(x.fun)

        self.result = np.array(self.result)
        self.C = self.result[:, 0]
        self.x = self.result[:, 1]*100
        chi = self.k / (self.k * self.C + 1)
        self.list_C45 = (1 - np.sqrt(1 - 4 * chi ** 2 * self.x / 100 * (1 - self.x/100) * self.C ** 2)) / 2 / chi
        self.list_C4 = (1-self.x/100) * self.C - self.list_C45
        self.list_C5 = self.x / 100 * self.C - self.list_C45

    def plot_fit(self, index):
        plt.plot(self.wl, self.list_spectra[index], label="data")
        plt.plot(self.wl, self.e45 * self.list_C45[index] + self.e4 * self.list_C4[index] + self.e5 * self.list_C5[index] ** self.p, label="fit")
        plt.legend()
        plt.xlabel("Wavelength(nm)")

    def plot_fit_param(self, c4, c5, c45):
        plt.plot(
            self.wl, self.e4 * c4 + self.e5 * c5 ** self.p + self.e45 * c45)
        plt.xlabel("Wavelength(nm)")

