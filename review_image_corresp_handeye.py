#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import yaml
import os
import random

from utils.find_images import find_image_pairs

# Ruta base de las imágenes
base_path = "../rgbt-ped-detection/kaist_dataset_images/kaist-cvpr15/images"

os.makedirs('data', exist_ok=True)
# Archivo YAML para almacenar los puntos
points_file = "data/he_points.yaml"

# Inicializar datos de YAML
if os.path.exists(points_file):
    with open(points_file, 'r') as f:
        points_data = yaml.safe_load(f) or {}

    total_points = sum(len(data["lwir"]) for data in points_data.values())

    print(f"[INFO] Archivo YAML cargado con {len(points_data)} imágenes procesadas previamente y {total_points} pares de puntos marcados.")
else:
    points_data = {}
    print(f"[INFO] No se encontró un archivo YAML previo. Se creará uno nuevo.")



# Encontrar pares
pairs = find_image_pairs(base_path)
random.shuffle(pairs)  # Barajar los pares para selección aleatoria

# Inicialización de variables
current_points_rgb = []
current_points_lwir = []
current_pair_index = 0

# Función de callback para capturar puntos en RGB
def select_points_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points_rgb.append((x, y))
        print(f"[INFO] Punto RGB añadido: {x}, {y}")

# Función de callback para capturar puntos en LWIR
def select_points_lwir(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points_lwir.append((x, y))
        print(f"[INFO] Punto LWIR añadido: {x}, {y}")

# Guardar puntos en YAML
# Guardar puntos en YAML
def save_points(lwir_file, visible_file):
    global current_points_rgb, current_points_lwir, points_data

    if len(current_points_rgb) != len(current_points_lwir):
        print("[ERROR] El número de puntos RGB y LWIR no coincide. Completa antes de continuar.")
        return False

    # Convertir las tuplas a listas para YAML
    points_data[lwir_file] = {
        "rgb_file": visible_file,
        "rgb": [list(point) for point in current_points_rgb],
        "lwir": [list(point) for point in current_points_lwir]
    }

    # Guardar en archivo
    with open(points_file, 'w') as f:
        yaml.dump(points_data, f)
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
    lwir_image_path = os.path.join(base_path, os.path.relpath(lwir_file, '/'))
    visible_image_path = os.path.join(base_path, os.path.relpath(visible_file, '/'))
    print(f"Images from basepath ({base_path}):\n\t· {lwir_image_path}.\n\t· {visible_image_path}.")
    lwir_image = cv2.imread(lwir_image_path, cv2.IMREAD_GRAYSCALE)
    visible_image = cv2.imread(visible_image_path)
    if lwir_image is None or visible_image is None:
        print(f"[ERROR] No se pudo cargar una o ambas imágenes: {lwir_file}, {visible_file}")
        current_pair_index += 1
        continue

    lwir_image = cv2.resize(lwir_image, (visible_image.shape[1], visible_image.shape[0]))  # Ajustar tamaños

    while True:
        # Mostrar ambas imágenes
        cv2.imshow('RGB', visible_image)
        cv2.imshow('LWIR', lwir_image)

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
            with open(points_file, 'w') as f:
                yaml.dump(points_data, f)
            total_points = sum(len(data["lwir"]) for data in points_data.values())
            print(f"[INFO] Guardando progreso y saliendo del programa. Procesadas {len(points_data)} imágenes y {total_points} puntos.")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
