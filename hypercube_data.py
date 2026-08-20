import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.signal import savgol_filter
from sklearn import preprocessing
import uuid
import datetime
import os
import pickle
import socket

# Example:Plot image
# from hypercube_data import cube
# im = cube(
#                        filename, wavearea=100, Firstnm=0, Lastnm=100).RGB_Image()
#                    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Example: get sepctrum data
# from hypercube_data import Cube_Read

# spectrum_data, pixel = Cube_Read(fileDat_Name, wavearea=100,
#                                                                                 Firstnm=0,
#                                                                                 Lastnm=100).cube_matrix()


class Cube_Read(object):
    def __init__(self, file_address, wavearea, Firstnm, Lastnm):
        self.file_address = file_address
        self.wavearea = wavearea
        self.Firstnm = Firstnm
        self.Lastnm = Lastnm

    def read_cube_dimension(file_adress):

        dim = np.fromfile(file_adress, dtype=">i4", count=3)

        return dim

    def read_cube(self):
        data_type = np.dtype("float32").newbyteorder(">")
        x = np.fromfile(self.file_address, dtype=data_type)

        spectrum = x[3:]  # eliminating the 3 leading zeros

        # eliminate negative values
        spectrum[spectrum < 0.00001] = 0.00001
        # eliminate huge values (assume 10 looking at the graphs)
        spectrum[spectrum > 10] = 10

        return spectrum

    def cube_matrix(self):
        spectrum_data = self.read_cube()
        # reorganise the data as the origibal HSI cube
        pixelXSize = int(np.size(spectrum_data) / 100)
        pixelYSize = int(round(pixelXSize / 640))
        data = spectrum_data.reshape((640, pixelYSize, 100))
        # data = np.vstack([np.hstack(cell) for cell in spectrum_data])

        return np.rot90(data[..., self.Firstnm : self.Lastnm + 1]), pixelYSize

    def cube_matrix_learn(self):
        spectrum_data = self.read_cube()
        # reorganise spectrum for classification test
        x = int(np.size(spectrum_data) / 100)
        data = spectrum_data.reshape((x, 100))
        return data, x

    def cube_SG_learn(self):
        spectrum_data = self.read_cube()
        pixelXSize = int(np.size(spectrum_data) / 100)
        data = spectrum_data.reshape((pixelXSize, 100))
        data_SG = savgol_filter(data, 9, 2, mode="nearest", axis=1)
        return data_SG

    def cube_SNV_learn(self):
        spectrum_data = self.cube_SG_learn()
        data_SNV = preprocessing.scale(spectrum_data, axis=1)
        return data_SNV

    def cube_snv_matrix(self):
        spectrum, pixely = self.cube_matrix_learn()
        mean = np.mean(spectrum, axis=0)
        print(mean.size)
        std = np.std(spectrum, axis=0)
        pixelXSize = int(np.size(spectrum) / 100)
        # data = np.zeros((307200,100))
        # data = np.zeros((307200, 80))
        data = np.zeros((pixelXSize, 100))
        print(mean.shape)
        for n in range(1, 100):
            for i in range(pixelXSize):
                data[i, n] = (spectrum[i, n] - mean[n]) / std[n]
        data = data[:, self.Firstnm : self.Lastnm]
        return data, pixely


class cube(object):
    def __init__(self, address, wavearea, Firstnm, Lastnm):
        self.address = address
        self.wavearea = wavearea
        self.Firstnm = Firstnm
        self.Lastnm = Lastnm

    # https://stackoverflow.com/questions/1627376/how-do-i-extract-a-ieee-be-binary-file-embedded-in-a-zipfile

    def cube_plot(self, show_plot=False):
        # the plot is done following the guidance from Tivita TM
        # Dokumentation RGB-Image1.4.vi
        initial_cube, y = Cube_Read(
            self.address, self.wavearea, self.Firstnm, self.Lastnm
        ).cube_matrix()

        # pixel rgb values
        # Shape is (Height, Width, 3) -> (480, 640, 3)
        RGB_values = np.zeros((y, 640, 3), dtype=np.dtype("float32"))

        # --- FIX STARTS HERE ---

        # 1. Blue: take 530-560nm (indices 6:13)
        # No transpose needed. We average over the last axis (bands) to keep spatial (y, x) intact.
        RGB_values[:, :, 2] = initial_cube[:, :, 6:13].mean(axis=2) * 1.5

        # 2. Green: take 540-590nm (indices 8:19)
        RGB_values[:, :, 1] = initial_cube[:, :, 8:19].mean(axis=2) * 1.5

        # 3. Red: take 585-725nm (indices 17:46)
        RGB_values[:, :, 0] = initial_cube[:, :, 17:46].mean(axis=2) * 1.5

        # --- FIX ENDS HERE ---

        # for normalisation of pixels to be between (0,1)
        R_min = np.min(RGB_values[:, :, 0])
        R_max = np.max(RGB_values[:, :, 0])

        G_min = np.min(RGB_values[:, :, 1])
        G_max = np.max(RGB_values[:, :, 1])

        B_min = np.min(RGB_values[:, :, 2])
        B_max = np.max(RGB_values[:, :, 2])

        # scaled RGB values
        scaled_RGB = np.zeros((y, 640, 3), dtype=np.dtype("float32"))

        # Avoid division by zero if max == min
        if B_max > B_min:
            scaled_RGB[:, :, 2] = (RGB_values[:, :, 2] - B_min) / (B_max - B_min)
        if G_max > G_min:
            scaled_RGB[:, :, 1] = (RGB_values[:, :, 1] - G_min) / (G_max - G_min)
        if R_max > R_min:
            scaled_RGB[:, :, 0] = (RGB_values[:, :, 0] - R_min) / (R_max - R_min)

        # add the gamma-factor
        gamma_corrected = exposure.adjust_gamma(scaled_RGB, 0.5)

        if show_plot:
            # plot the rgb image
            imgplot = plt.figure()
            plt.grid(False)
            # Use flipud because HSI cameras often scan lines that result in an inverted Y-axis
            imgplot = plt.imshow(np.flipud(gamma_corrected))

        return gamma_corrected

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"

def track_execution(student_id=None):
    log_file = "utils/el.pkl"

    run_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_id": str(uuid.uuid4()),
        "student_id": student_id or "unknown",
        "ip_address": get_ip(),
    }

    # Load existing logs or create new list
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            logs = pickle.load(f)
    else:
        logs = []

    # Append new log entry
    logs.append(run_data)

    # Save logs back to file
    with open(log_file, "wb") as f:
        pickle.dump(logs, f)

    # print(f"Execution logged: {run_data}")

import getpass
track_execution(student_id=getpass.getuser())