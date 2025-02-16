#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import os

from utils.affine_transform_utils import scaleAffineTransform, invertAffineTransform
from utils.load_optical_flow import load_optical_flow

from constants import output_data_path

computed_scale_affine = 5

# Cargar las imágenes base
dataset_path = "/home/arvc/eeha/yolo_test_utils/kaist_labeled_images/"
lwir_image_path = "set00/V000/lwir/I01689_labeled.jpg"
visible_image_path = lwir_image_path.replace('lwir', 'visible')

# Check previous transformation
prev_visible_image_path = "set00/V000/visible/I01688_labeled.jpg"
# visible_image_path = visible_image_path

def update_overlay(alpha):
    lwir_image_eq = cv2.equalizeHist(lwir_image)
    overlay = cv2.addWeighted(
        visible_image, 1 - alpha,
        cv2.cvtColor(lwir_image_eq, cv2.COLOR_GRAY2RGB), alpha, 0
    )
    cv2.imshow('Visible + LWIR', overlay)

    lwir_corrected_eq = cv2.equalizeHist(lwir_corrected)
    overlay_corrected = cv2.addWeighted(
        visible_image, 1 - alpha,
        cv2.cvtColor(lwir_corrected_eq, cv2.COLOR_GRAY2RGB), alpha, 0
    )
    cv2.imshow('Visible + LWIR Corrected', overlay_corrected)


    overlay_previous = cv2.addWeighted(visible_image, 1 - alpha, prev_visible_image, alpha, 0)
    cv2.imshow('Previous', overlay_previous)

def on_trackbar(val):
    alpha = val / 100
    update_overlay(alpha)
    cv2.setTrackbarPos('Transparencia', 'Visible + LWIR', val)
    cv2.setTrackbarPos('Transparencia', 'Visible + LWIR Corrected', val)

if __name__ == '__main__':
    prev_visible_abs_image_path = os.path.join(dataset_path, prev_visible_image_path)
    visible_abs_image_path = os.path.join(dataset_path, visible_image_path)
    lwir_abs_image_path = os.path.join(dataset_path, lwir_image_path)

    prev_visible_image = cv2.imread(prev_visible_abs_image_path)
    visible_image = cv2.imread(visible_abs_image_path)
    lwir_image = cv2.imread(lwir_abs_image_path, cv2.IMREAD_GRAYSCALE)

    flow_data = load_optical_flow("/"+visible_image_path.replace('_labeled',''))
    visible2visible_transform = np.array(flow_data['oflow_visible']['transformation_matrix'])
    visible2lwir_transform = scaleAffineTransform(visible2visible_transform, computed_scale_affine)
    lwir2visible_transform = invertAffineTransform(visible2lwir_transform)

    h, w = lwir_image.shape
    lwir_corrected = cv2.warpAffine(lwir_image, lwir2visible_transform, (w, h))

    cv2.namedWindow('Visible + LWIR')
    cv2.namedWindow('Visible + LWIR Corrected')
    cv2.createTrackbar('Transparencia', 'Visible + LWIR', 50, 100, on_trackbar)
    cv2.createTrackbar('Transparencia', 'Visible + LWIR Corrected', 50, 100, on_trackbar)

    update_overlay(0.5)

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # Tecla 'Esc' para salir
            cv2.destroyAllWindows()
            break
