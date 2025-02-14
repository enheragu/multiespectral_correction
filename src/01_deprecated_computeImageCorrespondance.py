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
image_lwir_path = "images/lwir_I01689.png"
image_rgb_path = "images/visible_I01689.png"

def generateCorrectedImage(camera_matrix_est, dist_coeffs_est, image_path, crop_data, tag = '_corrected', plot = True):
    crop_x, crop_y, crop_w, crop_h = crop_data
    
    ## Get corrected images!
    image = cv2.imread(image_path)
    # image_corrected = cv2.undistort(image, camera_matrix_est, dist_coeffs_est)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix_est, dist_coeffs_est, (image.shape[1], image.shape[0]), alpha=0
    ) # New projection matrix

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix_est, dist_coeffs_est, None, new_camera_matrix,
        (image.shape[1], image.shape[0]), cv2.CV_32FC1
    )
    image_corrected = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    def cropImage(image, x, y, w, h):
        return image[y:y+h, x:x+w]
    
    image_cropped = cropImage(image_corrected, *roi)
    final_image = cropImage(image_cropped, crop_x, crop_y, crop_w, crop_h)
    
    if plot:
        os.makedirs('plot', exist_ok=True)
        corrected_path = f'{image_path.replace(".png", f"{tag}.png").replace("images","plot")}'
        cv2.imwrite(corrected_path, final_image)

    return final_image

def correctImage(filtered_image_points_lwir, filtered_image_points_rgb, crop_data, tag='_corrected'):
    calibrate_data = calibrateImagePair(filtered_image_points_lwir, filtered_image_points_rgb)
    camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points = calibrate_data
    mean_reprojection_error, error_list = computeReprojectionError(*calibrate_data, filtered_image_points_lwir, 'plot/histogram_reprojection_filtered.png', bin_size=2)
    generateCorrectedImage(camera_matrix_est, dist_coeffs_est, image_lwir_path, crop_data, tag)
    generateCorrectedImage(camera_matrix_est, dist_coeffs_est, image_rgb_path, crop_data, tag)
    return calibrate_data


def applyTransformation(calibration_matrix, dist_coeffs, crop_data, points):
    crop_x, crop_y, crop_w, crop_h = crop_data
    crop_matrix = np.array([
                [1, 0, -crop_x],
                [0, 1, -crop_y],
                [0, 0, 1]

                ], dtype='float32')
    
    transformation_matrix = np.dot(calibration_matrix, crop_matrix)

    points = np.array(points, dtype='float32')
    points = np.expand_dims(points, axis=1)
    undistorted_points = cv2.undistortPoints(points, calibration_matrix, dist_coeffs)
    undistorted_points = cv2.perspectiveTransform(undistorted_points, transformation_matrix)

    return undistorted_points

def shadeRegion(image, x, y, w, h, shade_color=(90, 90, 90), alpha=0.5):
    shaded_image = image.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], y), shade_color, -1)
    cv2.rectangle(overlay, (0, y + h), (image.shape[1], image.shape[0]), shade_color, -1)
    cv2.rectangle(overlay, (0, y), (x, y + h), shade_color, -1)
    cv2.rectangle(overlay, (x + w, y), (image.shape[1], y + h), shade_color, -1)
    cv2.addWeighted(overlay, alpha, shaded_image, 1 - alpha, 0, shaded_image)
    return shaded_image

## GET POINTS FROM YAML FILE
image_points_lwir, image_points_rgb = getPointData()
plotDensityMap(image_points_lwir, image_points_rgb, tag = "_00_rawdata")

## CALIBRATE PAIR AND FILTER DATA BASED ON REPROJECTION ERROR
filtered_data = filterVectorsWithCalibration(image_points_lwir, image_points_rgb)
filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2 = filtered_data
plotDensityMap(filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2, image_size=IMAGE_SIZE, tag = "_01_calibrationFiltered")

filtered_image_points_lwir, filtered_image_points_rgb = filterVectorsWithNeighbours(filtered_image_points_lwir, filtered_image_points_rgb)
arrow_weigthed_filtered, _ = plotDensityMap(filtered_image_points_lwir, filtered_image_points_rgb, image_size=IMAGE_SIZE, tag = "_02_weightedFiltered")

grid_size = 80
# nn_filtered_image_points_lwir, nn_filtered_image_points_rgb = interpolateDistortionGridNN(filtered_image_points_lwir, filtered_image_points_rgb, image_size=IMAGE_SIZE, grid_size=grid_size)
# plotDensityMap(nn_filtered_image_points_lwir, nn_filtered_image_points_rgb, image_size=IMAGE_SIZE, tag = "_03_nnInterpolation")

# linear_filtered_image_points_lwir, linear_filtered_image_points_rgb = interpolateDistortionGridLinear(filtered_image_points_lwir, filtered_image_points_rgb, image_size=IMAGE_SIZE, grid_size=grid_size)
# plotDensityMap(linear_filtered_image_points_lwir, linear_filtered_image_points_rgb, image_size=IMAGE_SIZE, tag = "_04_linearInterpolation")

## CALIBRATE AGAIN BASED ON FILTERED DATA AND LOG
crop_x, crop_y, crop_w, crop_h = 80, 30, 550, 440
crop_data = crop_x, crop_y, crop_w, crop_h
calibrate_data = correctImage(filtered_image_points_lwir, filtered_image_points_rgb, crop_data, tag='_corrected')
# correctImage(linear_filtered_image_points_lwir, linear_filtered_image_points_rgb, tag='_linearinterpolated')

# Crop image and adapt labels?¿
camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points = calibrate_data
# image_lwir_corrected, image_lwir_cropped = applyTransformation(camera_matrix_est, dist_coeffs_est, crop_data)
# image_rgb_corrected, image_rgb_cropped = applyTransformation(camera_matrix_est, dist_coeffs_est, crop_data)


print(camera_matrix_est)
print(dist_coeffs_est)
arrow_weigthed_filtered_shaded = shadeRegion(arrow_weigthed_filtered, crop_x, crop_y, crop_w, crop_h)
cv2.imwrite("plot/calib_arrows_02_weightedFiltered_shadowed.png", arrow_weigthed_filtered_shaded)

cv2.pollKey()
# plt.show()
cv2.destroyAllWindows()