import sys
sys.path.extend(['/home/ange/MEGA/Python projects/Postdoc UC3M/SOC_measurement'])
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import rcParams
import numpy as np
import glob
from AS7341Reader import AS7341Reader
from sklearn.metrics import root_mean_squared_error
from matplotlib import rc
import re

#
# Enable LaTeX
rc('text', usetex=True)
rc('font', family='serif')
fontsize = 18
plt.rcParams.update({'font.size': fontsize})
rc('axes', titlesize=fontsize, labelsize=fontsize)  # Set title and label font size
rc('xtick', labelsize=fontsize)  # X-axis ticks
rc('ytick', labelsize=fontsize)  # Y-axis ticks
rc('legend', fontsize=fontsize)  # Legend font size

"""======================Graph spectra Figure 3"""

legend_params = {
    'loc': 'center',
    'bbox_to_anchor': (0.5, 0.95),
    'ncol': 6,
    'fontsize': 'medium',
    'handlelength': 2.5,
    'handletextpad': 0.2,
    'columnspacing': 1,
    'frameon': False,
    "labelspacing": 0.3
}


root = "Database/data_neg_1_8_M"
path_ref_18 = glob.glob(root + "/ref.csv")[0]
path_data_18 = glob.glob(root + "/150*")
path_dark_18 = glob.glob(root + "/dark.csv")[0]
path_data_18.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
r_neg = AS7341Reader(path_data_18, path_ref_18, path_dark_18, None, 0.015, None)

SOC = [0, 20, 40, 60, 80, 100]
colors = ["black", "blue", "green", "orange", "red", "purple"]



indices = []
fig, axs = plt.subplots(1,2, figsize=(15, 6))
plt.sca(axs[0])
for i, soc in enumerate(SOC):
    plt.plot(r_neg.abs[i*2].loc[445:630], color= colors[i], label=str(soc) + r"$\%$", marker="o")
fig.legend(**legend_params)
plt.xlabel("Wavelength(nm)")
plt.ylabel(r"Absorbance ($\mathrm{cm^{-1}}$)")
plt.title("NEG")

root = "Database/data_pos_1_8_M"
path_ref_18 = glob.glob(root + "/ref.csv")[0]
path_data_18 = glob.glob(root + "/150*")
path_dark_18 = glob.glob(root + "/dark.csv")[0]
path_data_18.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
r_pos = AS7341Reader(path_data_18, path_ref_18, path_dark_18, None, 0.015, None)

plt.sca(axs[1])
axs[1].yaxis.tick_right()
axs[1].yaxis.set_label_position("right")
for i, soc in enumerate(SOC):
    plt.plot(r_pos.abs[i*2].loc[445:630], color= colors[i], label=f"${soc}$", marker="o")
plt.xlabel("Wavelength(nm)")
plt.ylabel(r"Absorbance ($\mathrm{cm^{-1}}$)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.title("POS")



"""=====================================Graph deconvolution Figure 4============================"""
# legend_params = {
#     'loc': 'upper center',
#     'bbox_to_anchor': (0.5, 1.02),
#     'ncol': 2,
#     'fontsize': 'medium',
#     'handlelength': 2.5,
#     'handletextpad': 0.2,
#     'columnspacing': 1,
#     'frameon': False,
#     "labelspacing": 0.3
# }
# SOC = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# C_total = 1.83
# C_real = [C_total * 1.2 / 1.8, C_total * 1.5 / 1.8, C_total]
# SOC_long = np.concatenate((SOC, SOC, SOC))
# C = np.array(C_real)
# C_long = np.zeros(11)
# C_long = [C_long + C_0 for C_0 in C]
# C_long = np.asarray(C_long).reshape(-1)
#
# cal = "calibration_folders/cal_neg/cal_neg_deconvolution_AS7341_paper/"
# root = "Database/data_neg_1_8_M"
# path_ref_18 = glob.glob(root + "/ref.csv")[0]
# path_data_18 = glob.glob(root + "/150*")
# path_dark_18 = glob.glob(root + "/dark.csv")[0]
# path_data_18.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
# r_18 = AS7341Reader(path_data_18, path_ref_18, path_dark_18, cal, 0.015, None)
#
# root = "Database/data_neg_1_2_M"
# path_ref_12 = glob.glob(root + "/ref.csv")[0]
# path_data_12 = glob.glob(root + "/150*")
# path_dark_12 = glob.glob(root + "/dark.csv")[0]
# path_data_12.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
# r_12 = AS7341Reader(path_data_12, path_ref_12, path_dark_12, cal, 0.015, None)
#
# root = "Database/data_neg_1_5_M"
# path_ref_15 = glob.glob(root + "/ref.csv")[0]
# path_data_15 = glob.glob(root + "/150*")
# path_dark_15 = glob.glob(root + "/dark.csv")[0]
# path_data_15.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
# r_15 = AS7341Reader(path_data_15, path_ref_15, path_dark_15, cal, 0.015, None)
#
#
# r_18.calculate_soc()
# r_15.calculate_soc()
# r_12.calculate_soc()
# wl= r_18.preprocessing.wl[1:]
# idx = 5
# c = r_18.preprocessing.C[idx]
# soc = r_18.preprocessing.x[idx]
# e2 = r_18.preprocessing.e2[1:]
# e3 = r_18.preprocessing.e3[1:]
# y = r_18.preprocessing.list_y[idx]
# fig = plt.figure(figsize=(14, 10))
# gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
# ax1 = fig.add_subplot(gs[0,0:2])
# ax1.text(0.02, 1.1, "(a)", transform=ax1.transAxes,
#              fontweight="bold", va="top", ha="right", fontsize=23)
# ax1.text(600, 5.8, r"$\mathrm{V^{III}}$", color="green", fontsize=23)
# ax1.text(530, 4.7, r"$\mathrm{V^{II}}$", color="purple", fontsize=23)
# plt.plot(r_18.abs[idx].loc[445:680], color="black", marker="o", label="experiment")
# plt.plot(wl, e3 * (100 - SOC[idx]) / 100 * c+ e2 * SOC[idx] / 100 * c + y, label="fit", color="red", marker="o")
# plt.plot(wl,e3 * (100 - SOC[idx])/100*c, color="green", marker="o", linestyle="--")
# plt.plot(wl,e2*c*soc/100, color="purple", marker="o", linestyle="--")
# fig.legend(**legend_params)
# plt.xlabel("Wavelength(nm)")
# plt.ylabel(r"Absorbance ($\mathrm{cm^{-1}}$)")
#
# soc_p = []
# C_p = []
# for r in [r_12, r_15, r_18]:
#     C_p.append(r.data_c["C"].to_numpy())
#     soc_p.append(r.data_soc["SOC"].to_numpy())
# C_p = np.asarray(C_p).reshape(-1)
# soc_p = np.asarray(soc_p).reshape(-1)
# # Smaller plots in the second row
# ax2 = fig.add_subplot(gs[1, 0])  # Use the second row, first column
# ax2.text(0.1, 1.2, "(c)", transform=ax2.transAxes,
#              fontweight="bold", va="top", ha="right", fontsize=23)
# ax2.plot([0, 2], [0, 2], color="blue")
# ax2.scatter(C_long, C_p, s=10, color="black")
#
# plt.xlabel(r"$\mathrm{C_T(\%)}$")
# plt.ylabel(r"$\mathrm{C_P (\%)}$")
# plt.ylim(1.1,2)
# plt.xlim(1.1,2)
# rmse_C = root_mean_squared_error(C_long, C_p)
#
#
# ax3 = fig.add_subplot(gs[1, 1])  # Use the second row, second column
# ax3.text(0.1, 1.2, "(d)", transform=ax3.transAxes,
#              fontweight="bold", va="top", ha="right", fontsize=23)
# ax3.plot([0, 100], [0, 100], color="blue")
# ax3.scatter(SOC_long, soc_p, s=10, color="black")
# plt.xlabel(r"$\mathrm{SoC_T(\%)}$")
# plt.ylabel(r"$\mathrm{SoC_P (\%)}$")
# plt.ylim(0,100)
# plt.xlim(0,100)
# rmse_SOC = root_mean_squared_error(SOC_long, soc_p)
#
#
# cal = "calibration_folders/cal_pos/cal_pos_deconvolution/"
# root = "Database/data_pos_1_8_M"
# path_ref_18 = glob.glob(root + "/ref.csv")[0]
# path_dark_18 = glob.glob(root + "/dark.csv")[0]
# path_data_18 = glob.glob(root + "/150*")
# path_data_18.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
# r_18 = AS7341Reader(path_data_18, path_ref_18, path_dark_18, cal, 0.015, None)
#
# root = "Database/data_pos_1_2_M"
# path_ref_12 = glob.glob(root + "/ref.csv")[0]
# path_dark_12 = glob.glob(root + "/dark.csv")[0]
# path_data_12 = glob.glob(root + "/150*")
# path_data_12.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
# r_12 = AS7341Reader(path_data_12, path_ref_12, path_dark_12, cal, 0.015, None)
#
# root = "Database/data_pos_1_5_M"
# path_ref_15 = glob.glob(root + "/ref.csv")[0]
# path_dark_15 = glob.glob(root + "/dark.csv")[0]
# path_data_15 = glob.glob(root + "/150*")
# path_data_15.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
# r_15 = AS7341Reader(path_data_15, path_ref_15, path_dark_15, cal, 0.015, None)
#
# r_18.calculate_soc()
# r_15.calculate_soc()
# r_12.calculate_soc()
# wl= r_18.preprocessing.wl[1:-1]
#
# c = r_18.preprocessing.C[idx]
# c4 = r_18.preprocessing.list_C4[idx]
# c5 = r_18.preprocessing.list_C5[idx]
# c45 = r_18.preprocessing.list_C45[idx]
# soc = r_18.preprocessing.x[idx]
# e4 = r_18.preprocessing.e4[1:-1]
# e5 = r_18.preprocessing.e5[1:-1]
# e45 = r_18.preprocessing.e45[1:-1]
# ax4 = fig.add_subplot(gs[0,2:4])
# ax4.text(0.02, 1.1, "(b)", transform=ax4.transAxes,
#              fontweight="bold", va="top", ha="right", fontsize=23)
# ax4.text(470, 10, r"$\mathrm{V^{V}}$", color="orange", fontsize=23)
# ax4.text(600, 4.7, r"$\mathrm{V^{IV}}$", color="blue", fontsize=23)
# ax4.text(555, 50, r"$\mathrm V_2O_3^{3+}$", color="violet", fontsize=23)
#
# plt.plot(r_18.abs[idx].loc[445:630], color="black", marker="o", label="experiment")
# plt.plot(wl, e45 * c45 + e4 * c4 + e5 * c5 ** r_18.preprocessing.p, label="fit", color="red", marker="o")
# plt.plot(wl,e4 * c4, color="blue", marker="o", linestyle="--")
# plt.plot(wl,e5 * c5 ** r_18.preprocessing.p, color="orange", marker="o", linestyle="--")
# plt.plot(wl, e45 * c45, color="violet", marker="o", linestyle="--")
# plt.xlabel("Wavelength(nm)")
# plt.ylabel(r"Absorbance ($\mathrm{cm^{-1}}$)")
#
#
# soc_p = []
# C_p = []
# for r in [r_12, r_15, r_18]:
#     C_p.append(r.data_c["C"].to_numpy())
#     soc_p.append(r.data_soc["SOC"].to_numpy())
# C_p = np.asarray(C_p).reshape(-1)
# soc_p = np.asarray(soc_p).reshape(-1)
# # Smaller plots in the second row
# ax2 = fig.add_subplot(gs[1, 2])  # Use the second row, first column
# ax2.text(0.1, 1.2, "(e)", transform=ax2.transAxes,
#              fontweight="bold", va="top", ha="right", fontsize=23)
# ax2.plot([0, 2], [0, 2], color="blue")
# ax2.scatter(C_long, C_p, s=10, color="black")
#
# plt.xlabel(r"$\mathrm{C_T(\%)}$")
# plt.ylabel(r"$\mathrm{C_P (\%)}$")
# plt.ylim(1.1,2)
# plt.xlim(1.1,2)
# rmse_C = root_mean_squared_error(C_long, C_p)
#
#
# ax3 = fig.add_subplot(gs[1, 3])  # Use the second row, second column
# ax3.text(0.1, 1.2, "(f)", transform=ax3.transAxes,
#              fontweight="bold", va="top", ha="right", fontsize=23)
# ax3.plot([0, 100], [0, 100], color="blue")
# ax3.scatter(SOC_long, soc_p, s=10, color="black")
# plt.xlabel(r"$\mathrm{SoC_T(\%)}$")
# plt.ylabel(r"$\mathrm{SoC_P (\%)}$")
# plt.ylim(0,100)
# plt.xlim(0,100)
# rmse_SOC = root_mean_squared_error(SOC_long, soc_p)
# plt.tight_layout()
