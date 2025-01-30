#!/usr/bin/env python3
# encoding: utf-8

import yaml
import numpy as np

def getPointData(yaml_file_path = 'data/he_points.yaml'):

    # Cargar las parejas de puntos desde el archivo YAML
    with open(yaml_file_path, 'r') as f:
        points_data = yaml.safe_load(f) or {}

    # Puntos correspondientes entre las dos imágenes
    image_points_lwir = []  # Puntos de la cámara LWIR (distorsionados)
    image_points_rgb = []  # Puntos de la cámara RGB (corregidos)

    # Recorrer las imágenes y extraer los puntos correspondientes
    for image_path, data in points_data.items():
        rgb_points = data['rgb']
        lwir_points = data['lwir']
        
        for rgb_point, lwir_point in zip(rgb_points, lwir_points):
            image_points_lwir.append(lwir_point)
            image_points_rgb.append(rgb_point)

    # Convertir a formato adecuado para cv2.calibrateCamera
    image_points_lwir = np.array(image_points_lwir, dtype=np.float32)
    image_points_rgb = np.array(image_points_rgb, dtype=np.float32)

    return image_points_lwir, image_points_rgb