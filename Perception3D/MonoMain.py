# -*- coding: utf-8 -*-
from Calibration import MonoCalibration
from Rectification import MonoRectification

# Acquisition
calib = MonoCalibration(patternType="asymmetric_circles")
#calib.acquire(2)

# Calibration
rms, cameraMatrix, discoef, imgsize = calib.calibrate()

# Visualization
calib.visualizeBoards()
calib.plotRMS()

# Rectification
rectif = MonoRectification(calib.cameraMatrix, calib.distCoeffs, imgsize)
rectif.computeCorrectionMaps()
rectif.display(cameraId=2)

   