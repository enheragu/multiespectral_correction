#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from utils.pickle_utils import load_pkl
from constants import output_data_path

def getPointData(pkl_file_path = f'{output_data_path}/he_points.pkl'):

    points_data = load_pkl(pkl_file_path)

    image_points_lwir = []  # Distorted points
    image_points_rgb = []  # Correct points

    for image_path, data in points_data.items():
        rgb_points = data['rgb']
        lwir_points = data['lwir']
        
        for rgb_point, lwir_point in zip(rgb_points, lwir_points):
            image_points_lwir.append(lwir_point)
            image_points_rgb.append(rgb_point)

    # Convert to array for later camera calibration
    image_points_lwir = np.array(image_points_lwir, dtype=np.float32)
    image_points_rgb = np.array(image_points_rgb, dtype=np.float32)

    return image_points_lwir, image_points_rgb