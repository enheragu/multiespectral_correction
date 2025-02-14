#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import os
import random

from constants import output_data_path, dataset_images_path
from utils.find_images import find_image_pairs
from utils.pickle_utils import save_pkl, load_pkl

# Factor de escala para mostrar imágenes más grandes
scale_factor = 1.5

os.makedirs(output_data_path, exist_ok=True)
# Archivo Pickle para almacenar los puntos
points_file = f"{output_data_path}/he_points.pkl"

# Inicializar datos de Pickle
if os.path.exists(points_file):
    points_data = load_pkl(points_file)
    total_points = sum(len(data["lwir"]) for data in points_data.values())

    print(f"[INFO] Archivo Pickle cargado con {len(points_data)} imágenes procesadas previamente y {total_points} pares de puntos marcados.")
else:
    points_data = {}
    print(f"[INFO] No se encontró un archivo Pickle previo. Se creará uno nuevo.")


def resize_for_display(image, scale):
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

# Encontrar pares
pairs = find_image_pairs(dataset_images_path)
random.shuffle(pairs)  # Barajar los pares para selección aleatoria

# Inicialización de variables
current_points_rgb = []
current_points_lwir = []
current_pair_index = 0

# Función de callback para capturar puntos en RGB
def select_points_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)
        current_points_rgb.append((original_x, original_y))
        print(f"[INFO] Punto RGB añadido: {current_points_rgb[-1]}")

# Función de callback para capturar puntos en LWIR
def select_points_lwir(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)
        current_points_lwir.append((original_x, original_y))
        print(f"[INFO] Punto LWIR añadido: {current_points_lwir[-1]}")

# Guardar puntos en Pickle
def save_points(lwir_file, visible_file):
    global current_points_rgb, current_points_lwir, points_data

    if len(current_points_rgb) != len(current_points_lwir):
        print("[ERROR] El número de puntos RGB y LWIR no coincide. Completa antes de continuar.")
        print(f"{len(current_points_rgb) = }; {len(current_points_lwir) = }")
        return False

    # Convertir las tuplas a listas para Pickle
    points_data[lwir_file] = {
        "rgb_file": visible_file,
        "rgb": [list(point) for point in current_points_rgb],
        "lwir": [list(point) for point in current_points_lwir]
    }

    # Guardar en archivo
    save_pkl(points_data, points_file)
    print(f"[INFO] Puntos guardados para {lwir_file} y {visible_file}")
    return True


# Mostrar imágenes y permitir interacción
cv2.namedWindow('RGB')
cv2.namedWindow('LWIR')
cv2.setMouseCallback('RGB', select_points_rgb)
cv2.setMouseCallback('LWIR', select_points_lwir)

while current_pair_index < len(pairs):
    lwir_file, visible_file = pairs[current_pair_index]
    print(f"[INFO] Procesando par: LWIR={lwir_file}, Visible={visible_file}")

    # Evitar repetir imágenes ya procesadas
    if lwir_file in points_data:
        print(f"[INFO] Saltando {lwir_file}, ya tiene puntos marcados.")
        current_pair_index += 1
        continue

    # Cargar imágenes
    lwir_image_path = os.path.join(dataset_images_path, os.path.relpath(lwir_file, '/'))
    visible_image_path = os.path.join(dataset_images_path, os.path.relpath(visible_file, '/'))
    print(f"Images from basepath ({dataset_images_path}):\n\t· {lwir_image_path}.\n\t· {visible_image_path}.")
    lwir_image = cv2.imread(lwir_image_path, cv2.IMREAD_GRAYSCALE)
    visible_image = cv2.imread(visible_image_path)
    if lwir_image is None or visible_image is None:
        print(f"[ERROR] No se pudo cargar una o ambas imágenes: {lwir_file}, {visible_file}")
        current_pair_index += 1
        continue

    lwir_image = cv2.resize(lwir_image, (visible_image.shape[1], visible_image.shape[0]))  # Ajustar tamaños
    
    # Redimensionar imágenes para visualización
    visible_display = resize_for_display(visible_image, scale_factor)
    lwir_display = resize_for_display(lwir_image, scale_factor)

    while True:
        # Mostrar ambas imágenes
        cv2.imshow('RGB', visible_display)
        cv2.imshow('LWIR', lwir_display)

        key = cv2.waitKey(1)

        if key == ord('s'):  # Guardar puntos y pasar al siguiente par
            if save_points(lwir_file, visible_file):
                current_points_rgb = []
                current_points_lwir = []
                current_pair_index += 1
                print("[INFO] Pasando al siguiente par.")
                break
        
        elif key == ord('n'):  # Ignore image pair
            current_points_rgb = []
            current_points_lwir = []
            current_pair_index += 1
            print("[INFO] Pasando al siguiente par, este se descarta.")
            break

        elif key == 27:  # Esc para salir
            save_pkl(points_data, points_file)
            total_points = sum(len(data["lwir"]) for data in points_data.values())
            print(f"[INFO] Guardando progreso y saliendo del programa. Procesadas {len(points_data)} imágenes y {total_points} puntos.")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
