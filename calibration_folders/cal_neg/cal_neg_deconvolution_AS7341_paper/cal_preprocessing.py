import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize



def return_diff(x, spectrum, wl1, wl2, e3, e2):
    x_scaled = x.copy()
    # x_scaled[0] /= 100  # Normalize x[0] (e.g., between 0 and 1)
    # x_scaled[1] /= 10  # Normalize x[1] (e.g., between 0 and 10)
    spectrum = spectrum[wl1:wl2]
    e3 = e3[wl1:wl2]
    e2 = e2[wl1:wl2]
    diff = np.sum(abs(spectrum - (e3*(1-x[0])/1*x[1] + e2*x[0]/1*x[1])-x[2]))
    return diff

class PreprocessingSOC:

    """"Algorithm that performs deconvolution of the spectrum using epsilon 4,
    epsilon 5 and epsilon 45 (the complex absorbance)"""

    def __init__(self, list_spectra, calibration_folder, skip=1, arg=None):

        self.wl = [415, 445, 480, 515, 555, 590, 630, 680]
        self.list_spectra = []
        for spec in list_spectra:
            self.list_spectra.append(spec["Absorbance"].values)
        self.e3 = pd.read_csv(calibration_folder + "/e3.csv", index_col=0)
        self.e3 = self.e3["Absorbance"].values
        self.e2 = pd.read_csv(calibration_folder + "/e2.csv", index_col=0)
        self.e2 = self.e2["Absorbance"].values
        self.list_fit = []
        self.result = []
        self.index = []
        self.err = []
        self.success = []

        method = "L-BFGS-B"
        method = "SLSQP"
        options = {"gtol": 1e-15, "maxiter": 10000}
        for i, spec in enumerate(self.list_spectra):
            if divmod(i, skip)[1] == 0:
                x = minimize(return_diff, np.array([0.5, 1.5, 0]), args=(spec, 1, 7, self.e3, self.e2),
                             bounds=((0, 1), (1, 1.9), (-0.001, 0.001)), options=options, method=method)

                print(f"{i}/{len(self.list_spectra)}")
                print(x.success)
                print(x.x)
                self.result.append(x.x)
                self.list_fit.append(x)
                self.err.append(x.fun)
                self.success.append(x.success)

        self.result = np.array(self.result)
        self.C = self.result[:, 1]
        self.x = self.result[:, 0]*100
        self.list_y = self.result[:, 2]

    def plot_fit(self, index):
        plt.plot(self.wl, self.list_spectra[index], label="data")
        plt.plot(self.wl, self.e3 * (100-self.x[index]) / 100 * self.C[index] +
                 self.e2 * self.x[index] / 100 * self.C[index] + self.list_y[index], label="fit")
        plt.legend()
        plt.xlabel("Wavelength(nm)")

    def plot_fit_param(self, c2, c3):
        plt.plot(
            self.e3 * 0.01 * c3 + self.e2 * 0.01 *
            c2)
        plt.xlabel("Wavelength(nm)")

