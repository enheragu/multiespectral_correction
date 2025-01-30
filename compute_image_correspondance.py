#!/usr/bin/env python3
# encoding: utf-8

import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
import scipy.interpolate as interp

from utils.get_data import getPointData
from utils.filter_distortionvector_grid import filterVectorsWithCalibration, filterVectorsWithNeighbours, calibrateImagePair, computeReprojectionError
from utils.plot_utils import plotDensityMap 
from utils.nn_distortionspace_interpolation import interpolateDistortionGridNN
from utils.linear_distortionspace_interpolation import interpolateDistortionGridLinear

IMAGE_SIZE = 640,512
imagen_lwir_path = "images/lwir_I01689.png"
imagen_rgb_path = "images/visible_I01689.png"

def generateCorrectedImage(camera_matrix_est, dist_coeffs_est,rvecs,tvecs,tag = '_corrected'):
    ## Get corrected images!
    imagen_lwir = cv2.imread(imagen_lwir_path)
    imagen_rgb = cv2.imread(imagen_rgb_path)
    # imagen_lwir_corrected = cv2.undistort(imagen_lwir, camera_matrix_est, dist_coeffs_est)
    R, _ = cv2.Rodrigues(rvecs[0])  # rvecs[0] viene de la calibración
    T = tvecs[0]  # tvecs[0] también viene de la calibración
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix_est, dist_coeffs_est, (imagen_lwir.shape[1], imagen_lwir.shape[0]), alpha=0
    ) # New projection matrix

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix_est, dist_coeffs_est, None, new_camera_matrix,
        (imagen_lwir.shape[1], imagen_lwir.shape[0]), cv2.CV_32FC1
    )
    imagen_lwir_corrected = cv2.remap(imagen_lwir, map1, map2, interpolation=cv2.INTER_LINEAR)

    x, y, w, h = roi
    imagen_lwir_cropped = imagen_lwir_corrected[y:y+h, x:x+w]
    imagen_rgb_cropped = imagen_rgb[y:y+h, x:x+w]

    # cv2.imshow('LWIR Image Corrected', imagen_lwir_corrected)
    os.makedirs('plot', exist_ok=True)
    corrected_lwir_path = f'{imagen_lwir_path.replace(".png", f"{tag}.png").replace("images","plot")}'
    corrected_rgb_path = f'{imagen_rgb_path.replace(".png", f"{tag}.png").replace("images","plot")}'
    cv2.imwrite(corrected_lwir_path, imagen_lwir_cropped)
    cv2.imwrite(corrected_rgb_path, imagen_rgb_cropped)

    print(f"{corrected_lwir_path =}; {corrected_rgb_path =};")

def correctImage(filtered_image_points_lwir, filtered_image_points_rgb, tag='_corrected'):
    calibrate_data = calibrateImagePair(filtered_image_points_lwir, filtered_image_points_rgb)
    camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points = calibrate_data
    mean_reprojection_error, error_list = computeReprojectionError(*calibrate_data, filtered_image_points_lwir, 'plot/histogram_reprojection_filtered.png', bin_size=2)
    generateCorrectedImage(camera_matrix_est, dist_coeffs_est,rvecs,tvecs,tag)

## GET POINTS FROM YAML FILE
image_points_lwir, image_points_rgb = getPointData()
plotDensityMap(image_points_lwir, image_points_rgb, tag = "_00_rawdata")

## CALIBRATE PAIR AND FILTER DATA BASED ON REPROJECTION ERROR
filtered_data = filterVectorsWithCalibration(image_points_lwir, image_points_rgb)
filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2 = filtered_data
plotDensityMap(filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2, image_size=IMAGE_SIZE, tag = "_01_calibrationFiltered")

filtered_image_points_lwir, filtered_image_points_rgb = filterVectorsWithNeighbours(filtered_image_points_lwir, filtered_image_points_rgb)
plotDensityMap(filtered_image_points_lwir, filtered_image_points_rgb, image_size=IMAGE_SIZE, tag = "_02_weightedFiltered")

grid_size = 80
# nn_filtered_image_points_lwir, nn_filtered_image_points_rgb = interpolateDistortionGridNN(filtered_image_points_lwir, filtered_image_points_rgb, image_size=IMAGE_SIZE, grid_size=grid_size)
# plotDensityMap(nn_filtered_image_points_lwir, nn_filtered_image_points_rgb, image_size=IMAGE_SIZE, tag = "_03_nnInterpolation")

# linear_filtered_image_points_lwir, linear_filtered_image_points_rgb = interpolateDistortionGridLinear(filtered_image_points_lwir, filtered_image_points_rgb, image_size=IMAGE_SIZE, grid_size=grid_size)
# plotDensityMap(linear_filtered_image_points_lwir, linear_filtered_image_points_rgb, image_size=IMAGE_SIZE, tag = "_04_linearInterpolation")

## CALIBRATE AGAIN BASED ON FILTERED DATA AND LOG
correctImage(filtered_image_points_lwir, filtered_image_points_rgb, tag='_corrected')
# correctImage(linear_filtered_image_points_lwir, linear_filtered_image_points_rgb, tag='_linearinterpolated')


cv2.pollKey()
# plt.show()
cv2.destroyAllWindows()