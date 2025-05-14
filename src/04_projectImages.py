#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import os

from utils.affine_transform_utils import scaleAffineTransform, invertAffineTransform
from utils.load_optical_flow import load_optical_flow

from constants import output_data_path

scale_factor = 2
alpha = 50
# Cargar las imágenes base
dataset_path = "/home/arvc/eeha/yolo_test_utils/kaist_labeled_images/"
# lwir_image_path = "set00/V000/lwir/I01689_labeled.jpg" # Daylight example
# prev_visible_image_path = "set00/V000/visible/I01688_labeled.jpg" # Daylight example
lwir_image_path = "set01/V004/lwir/I00280_labeled.jpg" # Nightlight example
prev_visible_image_path = "set01/V004/visible/I00279_labeled.jpg" # Nightlight example
visible_image_path = lwir_image_path.replace('lwir', 'visible')

# Check previous transformation
# visible_image_path = visible_image_path

def update_overlay(alpha, scale_factor = scale_factor):
    visible2lwir_transform = scaleAffineTransform(visible2visible_transform, scale_factor)
    lwir2visible_transform = invertAffineTransform(visible2lwir_transform)
    
    h, w = lwir_image.shape
    lwir_corrected = cv2.warpAffine(lwir_image, lwir2visible_transform, (w, h))

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

    print(f"[update_overlay] {scale_factor = }")

def on_trackbar(val):
    global alpha, scale_factor
    alpha = val / 100
    update_overlay(alpha, scale_factor)
    cv2.setTrackbarPos('Transparencia', 'Visible + LWIR', val)
    cv2.setTrackbarPos('Transparencia', 'Visible + LWIR Corrected', val)

def on_trackbar_factor(val):
    global alpha, scale_factor
    scale_factor = val - 10
    update_overlay(alpha, scale_factor)
    cv2.setTrackbarPos('Transform factor', 'Visible + LWIR', val)
    cv2.setTrackbarPos('Transform factor', 'Visible + LWIR Corrected', val)

if __name__ == '__main__':
    prev_visible_abs_image_path = os.path.join(dataset_path, prev_visible_image_path)
    visible_abs_image_path = os.path.join(dataset_path, visible_image_path)
    lwir_abs_image_path = os.path.join(dataset_path, lwir_image_path)

    prev_visible_image = cv2.imread(prev_visible_abs_image_path)
    visible_image = cv2.imread(visible_abs_image_path)
    lwir_image = cv2.imread(lwir_abs_image_path, cv2.IMREAD_GRAYSCALE)

    flow_data = load_optical_flow("/"+visible_image_path.replace('_labeled',''))
    visible2visible_transform = np.array(flow_data['oflow_visible']['transformation_matrix'])

    cv2.namedWindow('Visible + LWIR')
    cv2.namedWindow('Visible + LWIR Corrected')
    cv2.createTrackbar('Transparencia', 'Visible + LWIR', 50, 100, on_trackbar)
    cv2.createTrackbar('Transparencia', 'Visible + LWIR Corrected', 50, 100, on_trackbar)

    cv2.createTrackbar('Transform factor', 'Visible + LWIR', 0, 10, on_trackbar_factor)
    cv2.createTrackbar('Transform factor', 'Visible + LWIR Corrected', 10, 20, on_trackbar_factor)

    update_overlay(0.5)

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # Tecla 'Esc' para salir
            cv2.destroyAllWindows()
            break
