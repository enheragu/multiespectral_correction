#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import os
import glob

# Cargar las imágenes base
lwir_image_path = 'images/lwir_I01689.png'
visible_image_path = lwir_image_path.replace('lwir', 'visible')
visible_image = cv2.imread(visible_image_path)
lwir_image = cv2.imread(lwir_image_path, cv2.IMREAD_GRAYSCALE)

print(f"{visible_image.shape = }; {lwir_image.shape = }")

# Función para buscar y cargar imágenes corregidas
def load_corrected_images(base_path, pattern):
    base_shape = visible_image.shape
    glob.glob(pattern)
    corrected_images = []
    corrected_files = glob.glob(pattern)
    for file in corrected_files:
        corrected_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if corrected_image.shape != base_shape:
            corrected_image = cv2.resize(corrected_image, (base_shape[1], base_shape[0]))
            corrected_images.append((file, corrected_image))
    return corrected_images

# Buscar todas las imágenes corregidas
pattern = os.path.join('plot', 'lwir_I01689_*.png')
corrected_images = load_corrected_images(lwir_image_path, pattern)
print(f"Check image projection with: {[data[0] for data in corrected_images]}")

# Función para actualizar la imagen mostrada con la superposición
def update_overlay(alpha):
    lwir_image_eq = cv2.equalizeHist(lwir_image)
    lwir_image_colored = lwir_image_eq

    overlay1 = visible_image.copy()
    overlay1 = cv2.addWeighted(
        overlay1, 1 - alpha,
        cv2.cvtColor(lwir_image_colored, cv2.COLOR_GRAY2RGB), alpha, 0
    )

    cv2.imshow('Visible + LWIR', overlay1)

    for file, image in corrected_images:
        lwir_image_corrected_eq = cv2.equalizeHist(image)
        lwir_image_corrected_colored = lwir_image_corrected_eq

        overlay2 = visible_image.copy()
        overlay2 = cv2.addWeighted(
            overlay2, 1 - alpha,
            cv2.cvtColor(lwir_image_corrected_colored, cv2.COLOR_GRAY2RGB), alpha, 0
        )

        window_name = os.path.basename(file).split('.')[0]
        cv2.imshow(f'Visible + {window_name}', overlay2)

# Callback para ajustar la transparencia, sincronizando ambos trackbars
def on_trackbar(val):
    alpha = val / 100
    update_overlay(alpha)
    cv2.setTrackbarPos('Transparencia', 'Visible + LWIR', val)
    for file, _ in corrected_images:
        window_name = os.path.basename(file).split('.')[0]
        cv2.setTrackbarPos('Transparencia', f'Visible + {window_name}', val)

alpha = 0.5
cv2.namedWindow('Visible + LWIR')
cv2.createTrackbar('Transparencia', 'Visible + LWIR', int(alpha * 100), 100, on_trackbar)
for file, _ in corrected_images:
    window_name = os.path.basename(file).split('.')[0]
    cv2.namedWindow(f'Visible + {window_name}')
    cv2.createTrackbar('Transparencia', f'Visible + {window_name}', int(alpha * 100), 100, on_trackbar)

update_overlay(alpha)

while True:
    key = cv2.waitKey(0)
    if key == 27:  # Tecla 'Esc' para salir
        cv2.destroyAllWindows()
        break
