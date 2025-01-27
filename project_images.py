#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np


# Cargar las imágenes
lwir_image_path = 'images/lwir_I01689.png'
lwir_image_corrected_path = 'plot/lwir_I01689_corrected.png'
visible_image_path = lwir_image_path.replace('lwir', 'visible')
visible_image_corrected_path = lwir_image_corrected_path.replace('lwir', 'visible')

visible_image = cv2.imread(visible_image_path)
visible_image_corrected = cv2.imread(visible_image_corrected_path)
lwir_image = cv2.imread(lwir_image_path, cv2.IMREAD_GRAYSCALE)
lwir_image_corrected = cv2.imread(lwir_image_corrected_path, cv2.IMREAD_GRAYSCALE)

print(f"{visible_image.shape = }; {visible_image_corrected.shape = }")
print(f"{lwir_image.shape = }; {lwir_image_corrected.shape = }")

# Función para actualizar la imagen mostrada con la superposición
def update_overlay():
    global alpha

    lwir_image_eq = cv2.equalizeHist(lwir_image)
    lwir_image_colored = lwir_image_eq #cv2.applyColorMap(lwir_image_eq, cv2.COLORMAP_JET)  # Usamos el colormap JET

    overlay1 = visible_image.copy()
    overlay1 = cv2.addWeighted(
        overlay1, 1 - alpha,
        cv2.cvtColor(lwir_image_colored, cv2.COLOR_BGR2RGB), alpha, 0
    )

    lwir_image_corrected_eq = cv2.equalizeHist(lwir_image_corrected)
    lwir_image_corrected_colored = lwir_image_corrected_eq #cv2.applyColorMap(lwir_image_corrected_eq, cv2.COLORMAP_JET)
    
    overlay2 = visible_image_corrected.copy()
    overlay2 = cv2.addWeighted(
        overlay2, 1 - alpha,
        cv2.cvtColor(lwir_image_corrected_colored, cv2.COLOR_BGR2RGB), alpha, 0
    )

    cv2.imshow('Visible + LWIR', overlay1)
    cv2.imshow('Visible + LWIR Corrected', overlay2)
    # print(f"Transparencia actual: {alpha}")

# Callback para ajustar la transparencia
def on_trackbar(val):
    global alpha
    alpha = val / 100
    update_overlay()

alpha = 0.5
cv2.namedWindow('Overlay')
cv2.createTrackbar('Transparencia', 'Overlay', int(alpha * 100), 100, on_trackbar)

update_overlay()

while True:
    key = cv2.waitKey(0)
    if key == 27:  # Tecla 'Esc' para salir
        cv2.destroyAllWindows()
        break
