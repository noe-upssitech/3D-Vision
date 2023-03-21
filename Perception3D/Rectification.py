# -*- coding: utf-8 -*-
import CalibUtils as cu
import cv2 as cv
import numpy as np

class MonoRectification:
    def __init__(self, cameraMatrix, distCoeffs, imageSize):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.imageSize = imageSize
        self.alpha = 1
        self.xRatio = 1
        self.yRatio = 1
    
    def computeCorrectionMaps(self, alpha = 1.0, xRatio = 1, yRatio = 1):
        newImageSize = (int(self.imageSize[0] / xRatio), int(self.imageSize[1] / yRatio))
        # Implement getOptimalNewCameraMatrix here
        newcameramtx, _ = cv.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, self.imageSize, alpha, newImageSize)
        # Implement initUndistortRectifyMap here
        self.map1, self.map2 = cv.initUndistortRectifyMap(self.cameraMatrix, self.distCoeffs, None, newcameramtx, self.imageSize, cv.CV_32FC1)
        
    def rectify(self, frame):
        return cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)
        
    def display(self, cameraId = 0, fps = 25):
        capture = cv.VideoCapture(cameraId)
        
        cv.namedWindow('Rectified Image', cv.WINDOW_NORMAL)
        
        # create trackbars
        cv.createTrackbar('10 * alpha', 'Rectified Image', 10 * self.alpha, 10, cu.nullFunction)
        cv.createTrackbar('xRatio', 'Rectified Image', self.xRatio, 5, cu.nullFunction)
        cv.createTrackbar('yRatio', 'Rectified Image', self.yRatio, 5, cu.nullFunction)
        self.computeCorrectionMaps(self.alpha, self.xRatio, self.yRatio)
        
        while(True):
            # Capture frame-by-frame
            ret, frame = capture.read()
            
            alpha = cv.getTrackbarPos('10 * alpha', 'Rectified Image') / 10
            xRatio = cv.getTrackbarPos('xRatio', 'Rectified Image')
            yRatio = cv.getTrackbarPos('yRatio', 'Rectified Image')
            
            if xRatio == 0:
                xRatio = 1
            if yRatio == 0:
                yRatio = 1
            
            if (alpha != self.alpha or xRatio != self.xRatio or yRatio != self.yRatio):
                self.alpha = alpha
                self.xRatio = xRatio
                self.yRatio = yRatio
                
                self.computeCorrectionMaps(self.alpha, self.xRatio, self.yRatio)
                
            frame = self.rectify(frame)
                
            cv.imshow('Rectified Image', frame)           
            
            key = cv.waitKey(int(1000 / fps))

            if key == ord('\x1b') or key == ord('q'):
                break
            
        capture.release()
        cv.destroyAllWindows()
        
class StereoRectification:
    def __init__(self, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, imageSize, R, T):
        self.cameraMatrixLeft = cameraMatrixLeft
        self.distCoeffsLeft = distCoeffsLeft
        self.cameraMatrixRight = cameraMatrixRight
        self.distCoeffsRight = distCoeffsRight
        self.imageSize = imageSize
        self.R = R
        self.T = T
        self.alpha = 1
        self.ratio = 1
        self.crop = 0
        self.epipolarLines = 0
    
    def computeCorrectionMaps(self, alpha = 1.0, ratio = 1):
        newImageSize = (int(self.imageSize[0] / ratio), int(self.imageSize[1] / ratio))
        
        # Implement stereoRectify here
        RLeft, PLeft, RRight, PRight = cv.stereoRectify(self.cameraMatrixLeft, self.distCoeffsLeft,
                                                        self.cameraMatrixRight, self.distCoeffsRight,
                                                        self.imageSize, self.R, self.T)

        self.mapxLeft, self.mapyLeft = cv.initUndistortRectifyMap(self.cameraMatrixLeft, self.distCoeffsLeft, RLeft, PLeft, newImageSize, cv.CV_32FC1)
        self.mapxRight, self.mapyRight = cv.initUndistortRectifyMap(self.cameraMatrixRight, self.distCoeffsRight, RRight, PRight, newImageSize, cv.CV_32FC1)
        
    def rectify(self, frameLeft, frameRight):
        left = cv.remap(frameLeft, self.mapxLeft, self.mapyLeft, cv.INTER_LINEAR)
        right = cv.remap(frameRight, self.mapxRight, self.mapyRight, cv.INTER_LINEAR)
        return left, right
        
    def display(self, left, right):
        cv.namedWindow('Rectified Stereo Pair', cv.WINDOW_NORMAL)
        
        # create trackbars
        cv.createTrackbar('10 * alpha', 'Rectified Stereo Pair', 10 * self.alpha, 10, cu.nullFunction)
        cv.createTrackbar('Ratio', 'Rectified Stereo Pair', self.ratio, 5, cu.nullFunction)
        cv.createTrackbar('Display N Epipolar lines', 'Rectified Stereo Pair', self.epipolarLines, 50, cu.nullFunction)
        self.computeCorrectionMaps(self.alpha, self.ratio)
        
        while(True):            
            alpha = cv.getTrackbarPos('10 * alpha', 'Rectified Stereo Pair') / 10
            ratio = cv.getTrackbarPos('Ratio', 'Rectified Stereo Pair')
            self.epipolarLines = cv.getTrackbarPos('Display N Epipolar lines', 'Rectified Stereo Pair')
            
            if ratio == 0:
                ratio = 1
                
            if (alpha != self.alpha or ratio != self.ratio):
                self.alpha = alpha
                self.ratio = ratio
                self.computeCorrectionMaps(self.alpha, self.ratio)
                
            rectLeft, rectRight = self.rectify(left, right)
            display = cv.cvtColor(cv.hconcat([rectLeft, rectRight]), cv.COLOR_GRAY2BGR)
            
            for i in range(0, self.epipolarLines):
                display = cv.line(display, (0, int(i*display.shape[1::-1][1]/self.epipolarLines)), (display.shape[1::-1][0]-1, int(i*display.shape[1::-1][1]/self.epipolarLines)), (0, 0, 200))
            
            cv.imshow('Rectified Stereo Pair', display)           
            
            key = cv.waitKey(1)
            
            if key == ord('\x1b') or key == ord('q'):
                break
            
        cv.destroyAllWindows()
        
    def displayDisparity(self, left, right):
        cv.namedWindow('Disparity', cv.WINDOW_NORMAL)
        
        # create trackbars
        cv.createTrackbar('10 * alpha', 'Disparity', 10 * self.alpha, 10, cu.nullFunction)
        cv.createTrackbar('Ratio', 'Disparity', self.ratio, 5, cu.nullFunction)
        cv.createTrackbar('numDisparities / 16', 'Disparity', 1, 20, cu.nullFunction)
        cv.createTrackbar('(blockSize - 5) / 2', 'Disparity', 8, 20, cu.nullFunction)
        cv.createTrackbar('uniquenessRatio', 'Disparity', 15, 50, cu.nullFunction)
        self.computeCorrectionMaps(self.alpha, self.ratio)
        
        while(True):            
            alpha = cv.getTrackbarPos('10 * alpha', 'Disparity') / 10
            ratio = cv.getTrackbarPos('Ratio', 'Disparity')
            numDisparities = 16 * cv.getTrackbarPos('numDisparities / 16', 'Disparity')
            blockSize = 2 * cv.getTrackbarPos('(blockSize - 5) / 2', 'Disparity') + 5
            uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'Disparity')
            
            if ratio == 0:
                ratio = 1
            
            if (alpha != self.alpha or ratio != self.ratio):
                self.alpha = alpha
                self.ratio = ratio
                self.computeCorrectionMaps(self.alpha, self.ratio)
                
            rectLeft, rectRight = self.rectify(left, right)
            
            # Construct the stereo object here (StereoBM_create)
            
            stereo.setMinDisparity(1)
            stereo.setUniquenessRatio(uniquenessRatio)
            
            # Compute the disparity here
            
            minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(disparity)
            display = cv.convertScaleAbs(disparity, alpha = (255.0 / maxVal - minVal))
            display = cv.cvtColor(display, cv.COLOR_GRAY2BGR)
            display = cv.applyColorMap(display, cv.COLORMAP_JET)
            
            mask = np.copy(disparity)
            mask[np.where(disparity <= [stereo.getNumDisparities() -16])] = [0]
            mask[np.where(disparity > [stereo.getNumDisparities() -16])] = [1]            
            mask = np.uint8(mask)
            display = cv.bitwise_and(display, display, mask = mask)
            
            cv.imshow('Disparity', display)           
            
            key = cv.waitKey(1)
            
            if key == ord('\x1b') or key == ord('q'):
                break
            
        cv.destroyAllWindows()
