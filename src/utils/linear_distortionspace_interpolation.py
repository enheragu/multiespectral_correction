#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
from scipy.spatial import cKDTree
import scipy.interpolate as interp


# Based on vectors (lwir to rgb points) average them in areas and interpolate to cover the whole image
def interpolateDistortionGridLinear(image_points_lwir, image_points_rgb, image_size, grid_size = 40):
    vectors = image_points_rgb - image_points_lwir
    
    grid_x, grid_y = np.meshgrid( # meshgrid is uniform in space
        np.linspace(0, image_size[0], grid_size),
        np.linspace(0, image_size[1], grid_size)
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    vector_field_x = interp.griddata(image_points_lwir, vectors[:, 0], (grid_x, grid_y), method='linear')
    vector_field_y = interp.griddata(image_points_lwir, vectors[:, 1], (grid_x, grid_y), method='linear')

    interpolated_lwir = grid_points
    interpolated_rgb = interpolated_lwir + np.column_stack(
        [vector_field_x.ravel(), vector_field_y.ravel()]
    )

    # Filter Nan data
    nan_indices = np.isnan(interpolated_rgb).any(axis=1)
    interpolated_lwir = interpolated_lwir[~nan_indices]
    interpolated_rgb = interpolated_rgb[~nan_indices]
    return np.array(interpolated_lwir, dtype='float32'), np.array(interpolated_rgb, dtype='float32')