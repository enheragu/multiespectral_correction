#!/usr/bin/env python3
# encoding: utf-8

import os
import cv2
import numpy as np
import yaml
import os
import random

from utils.find_images import find_image_pairs

# Ruta base de las imágenes
base_path = "../rgbt-ped-detection/kaist_dataset_images/kaist-cvpr15/images"

os.makedirs('data', exist_ok=True)

detector = cv2.ORB_create()
# detector = cv2.SIFT_create(nfeatures=350)
DETECTOR_NAME = detector.getDefaultName().lower().split('.')[-1]
points_file = f"data/{DETECTOR_NAME}_points.yaml"

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
matches_to_remove = set()

def filter_by_proximity(keypoints_lwir, keypoints_visible, matches, max_distance=70):
    filtered_matches = []
    for match in matches:
        kp1 = keypoints_lwir[match.queryIdx].pt  # Punto en la imagen LWIR
        kp2 = keypoints_visible[match.trainIdx].pt  # Punto en la imagen visible
        distance = np.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)

        if distance < max_distance:
            filtered_matches.append(match)
    return filtered_matches



def detect_and_match_features(lwir_image, visible_image):
    global detector, DETECTOR_NAME

    # Detectar y describir características
    keypoints_lwir, descriptors_lwir = detector.detectAndCompute(lwir_image, None)
    keypoints_visible, descriptors_visible = detector.detectAndCompute(visible_image, None)

    # Emparejar características con un matcher basado en Hamming
    if DETECTOR_NAME == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif DETECTOR_NAME == 'sift':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        print(f"Unknown detector: {DETECTOR_NAME}")
        exit()

    matches = matcher.match(descriptors_lwir, descriptors_visible)

    # Ordenar los matches por distancia (mejor primero)
    matches = sorted(matches, key=lambda x: x.distance)
    # Filter points by distance
    good_matches = filter_by_proximity(keypoints_lwir, keypoints_visible, matches)
    return keypoints_lwir, keypoints_visible, good_matches

def draw_matches_with_numbers(lwir_image, visible_image, keypoints_lwir, keypoints_visible, matches):
    result_image = np.hstack((visible_image, cv2.cvtColor(lwir_image, cv2.COLOR_GRAY2BGR)))
    h, w = lwir_image.shape

    for i, match in enumerate(matches):
        pt_lwir = tuple(map(int, keypoints_lwir[match.queryIdx].pt))
        pt_visible = tuple(map(int, keypoints_visible[match.trainIdx].pt))

        pt_visible_shifted = (pt_visible[0], pt_visible[1])
        pt_lwir_shifted = (pt_lwir[0] + w, pt_lwir[1])

        color = (0, 255, 0) if i not in matches_to_remove else (0, 0, 255)

        # Dibujar líneas
        cv2.line(result_image, pt_visible_shifted, pt_lwir_shifted, color, 1)
        
        # Dibujar números en los puntos
        cv2.putText(result_image, str(i), pt_visible_shifted, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(result_image, str(i), pt_lwir_shifted, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result_image

def save_points(lwir_file, visible_file, keypoints_lwir, keypoints_visible, matches):
    global points_data

    valid_matches = [m for i, m in enumerate(matches) if i not in matches_to_remove]
    if not valid_matches:
        print("[ERROR] No hay puntos válidos para guardar.")
        return False

    points_data[lwir_file] = {
        "rgb_file": visible_file,
        "rgb": [list(map(int, keypoints_visible[m.trainIdx].pt)) for m in valid_matches],
        "lwir": [list(map(int, keypoints_lwir[m.queryIdx].pt)) for m in valid_matches]
    }

    with open(points_file, 'w') as f:
        yaml.dump(points_data, f)

    print(f"[INFO] Guardados {len(valid_matches)} puntos para {lwir_file} y {visible_file}")
    return True

cv2.namedWindow('Matches')

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

    lwir_image = cv2.resize(lwir_image, (visible_image.shape[1], visible_image.shape[0]))

    # Detectar y emparejar características
    keypoints_lwir, keypoints_visible, matches = detect_and_match_features(lwir_image, visible_image)

    while True:
        result_image = draw_matches_with_numbers(lwir_image, visible_image, keypoints_lwir, keypoints_visible, matches)
        cv2.imshow('Matches', result_image)

        key = cv2.waitKey(1)

        if key == ord('s'):  # Guardar puntos y pasar al siguiente par
            if save_points(lwir_file, visible_file, keypoints_lwir, keypoints_visible, matches):
                matches_to_remove.clear()
                current_pair_index += 1
                print("[INFO] Pasando al siguiente par.")
                break

        elif key == 27:  # Esc para salir
            with open(points_file, 'w') as f:
                yaml.dump(points_data, f)
            print("[INFO] Guardando progreso y saliendo del programa.")
            cv2.destroyAllWindows()
            exit()

        elif key == ord('d'):  # Eliminar un punto
            print("[INFO] Presione el número del punto a eliminar.")
            idx = cv2.waitKey(0) - ord('0')
            if 0 <= idx < len(matches):
                matches_to_remove.add(idx)
                print(f"[INFO] Punto {idx} marcado para eliminar.")

        elif key == ord('n'):  # Ignor eimage
            matches_to_remove.clear()
            current_pair_index += 1
            print("[INFO] Pasando al siguiente par.")
            break

cv2.destroyAllWindows()
