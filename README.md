This is the raw data and the python code related to the paper:"Low-cost optical multi-wavelength sensor for accurate real-time state-of-charge monitoring in vanadium flow batteries"
from *Ange A. Maurice, Pablo A. Prieto-Díaz and Marcos Vera*

The data is acquired with samples ranging from 1.2 to 1.8 mole/L and from 0 to 100% State of Charge for the negolyte and the posolyte ($V^{II}/V^{III}$ and $V^{IV}/V^{V}$)

The data is within the Database folder: Every folder  correspond to a concentration and an electrolyte: data_neg_1_8_M is the negolyte at a concentration of 1.82 mole/L.

*dark.csv* is the spectrum acquired without light and ref.csv is the reference acquired with water.

*calibration_code.py*: contains the script used to plot Figure 3 and 4. It contains all the variables used for the calibration

*AS7341reader.py*: contains the class that reads a list of spectra and calculate the state of charge and concentration based on the provided calibration folder.
