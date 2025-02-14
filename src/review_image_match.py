#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np

# Función para actualizar la imagen mostrada con la superposición
def update_overlay():
    global x_offset, y_offset, camera_matrix, dist_coeffs

    # Crear una copia de la imagen RGB para superponer con tinte azul
    blue_rgb = rgb_image.copy()
    blue_rgb[:, :, 1] = 0  # Eliminamos el componente verde
    blue_rgb[:, :, 2] = 0  # Eliminamos el componente rojo

    # Aplicar CLAHE (Ecualización adaptativa de histograma) a la imagen LWIR
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_lwir = clahe.apply(gray_image)

    # Corregir distorsión en la imagen LWIR usando la matriz de cámara y coeficientes
    h, w = equalized_lwir.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    corrected_lwir = cv2.undistort(equalized_lwir, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crear la imagen LWIR con tinte amarillo (mezclamos rojo y verde)
    yellow_lwir = cv2.cvtColor(corrected_lwir, cv2.COLOR_GRAY2BGR)
    yellow_lwir[:, :, 0] = 0  # Eliminamos el componente azul (hacemos que sea amarillo)

    # Aseguramos que ambas imágenes tienen el mismo tamaño
    yellow_lwir_resized = cv2.resize(yellow_lwir, (rgb_width, rgb_height))

    # Mezclar ambos canales RGB y LWIR
    combined_image = np.zeros_like(blue_rgb)
    combined_image[:, :, 0] = yellow_lwir_resized[:, :, 1]  # Canal verde de LWIR (amarillo)
    combined_image[:, :, 1] = yellow_lwir_resized[:, :, 2]  # Canal rojo de LWIR (amarillo)
    combined_image[:, :, 2] = blue_rgb[:, :, 0]  # Canal azul de RGB

    # Redimensionar para que quepa en la pantalla con espacio para los trackbars
    max_width = 1280  # Tamaño máximo en ancho
    max_height = 720  # Tamaño máximo en altura
    height, width = combined_image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    resized_image = cv2.resize(combined_image, (int(width * scaling_factor), int(height * scaling_factor)))

    # Mostrar la imagen redimensionada
    cv2.imshow('Overlay', resized_image)

    # Mostrar parámetros de calibración en consola
    print("\n--- Parámetros de calibración actualizados ---")
    print(f"Desfase actual: ({x_offset - rgb_width // 2}, {y_offset - rgb_height // 2})")
    print("Matriz de la cámara (camera_matrix):")
    print(camera_matrix)
    print("Coeficientes de distorsión (dist_coeffs):")
    print(dist_coeffs)

# Callback para manejar los eventos de las teclas
def on_key(event):
    global x_offset, y_offset
    if event == 27:  # Tecla 'Esc' para salir
        cv2.destroyAllWindows()
        exit()
    elif event == ord('w'):  # Arriba
        y_offset -= 1
    elif event == ord('s'):  # Abajo
        y_offset += 1
    elif event == ord('a'):  # Izquierda
        x_offset -= 1
    elif event == ord('d'):  # Derecha
        x_offset += 1
    update_overlay()


# Cargar las imágenes
lwir_image_path = 'images/lwir_I01689.jpg'
visible_image_path = lwir_image_path.replace('lwir', 'visible')

rgb_image = cv2.imread(visible_image_path)  # Imagen en color
gray_image = cv2.imread(lwir_image_path, cv2.IMREAD_GRAYSCALE)  # Imagen en escala de grises

# Dimensiones de las imágenes
rgb_height, rgb_width = rgb_image.shape[:2]
gray_height, gray_width = gray_image.shape[:2]

# Centro inicial de la imagen en escala de grises
x_offset, y_offset = rgb_width // 2, rgb_height // 2

# Inicialización de los parámetros de la cámara y distorsión
# Inicialización de los parámetros de la cámara y distorsión
fx, fy = 765, 765
cx, cy = rgb_width // 2, rgb_height // 2
k1, k2, p1, p2, k3 = -0.1, 0.01, 0.0005, 0.0005, 0

# Inicializar la matriz de la cámara y los coeficientes de distorsión
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)



def on_distortion(val):
    global camera_matrix, dist_coeffs
    fx = cv2.getTrackbarPos('fx', 'Overlay')
    fy = cv2.getTrackbarPos('fy', 'Overlay')
    cx = cv2.getTrackbarPos('cx', 'Overlay')
    cy = cv2.getTrackbarPos('cy', 'Overlay')
    k1 = (cv2.getTrackbarPos('k1', 'Overlay') - 5) / 100
    k2 = (cv2.getTrackbarPos('k2', 'Overlay') - 50) / 1000
    p1 = (cv2.getTrackbarPos('p1', 'Overlay') - 50) / 10000
    p2 = (cv2.getTrackbarPos('p2', 'Overlay') - 50) / 10000
    k3 = (cv2.getTrackbarPos('k3', 'Overlay') - 50) / 1000

    # Actualizar la matriz de la cámara y los coeficientes de distorsión
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    update_overlay()

# Crear ventana
cv2.namedWindow('Overlay')

# Crear sliders para los parámetros de distorsión
cv2.createTrackbar('k1', 'Overlay', int((k1 + 0.5) * 100), 100, on_distortion)
cv2.setTrackbarPos('k1', 'Overlay', int((k1 + 0.05) * 1000))
cv2.createTrackbar('k2', 'Overlay', int((k2 + 0.05) * 1000), 100, on_distortion)
cv2.setTrackbarPos('k2', 'Overlay', int((k2 + 0.05) * 1000))
cv2.createTrackbar('p1', 'Overlay', int((p1 + 0.005) * 10000), 100, on_distortion)
cv2.setTrackbarPos('p1', 'Overlay', int((p1 + 0.005) * 10000))
cv2.createTrackbar('p2', 'Overlay', int((p2 + 0.005) * 10000), 100, on_distortion)
cv2.setTrackbarPos('p2', 'Overlay', int((p2 + 0.005) * 10000))
cv2.createTrackbar('k3', 'Overlay', int((k3 + 0.05) * 1000), 100, on_distortion)
cv2.setTrackbarPos('k3', 'Overlay', int((k3 + 0.05) * 1000))
cv2.createTrackbar('fx', 'Overlay', fx, 2000, on_distortion)
cv2.setTrackbarPos('fx', 'Overlay', fx)
cv2.createTrackbar('fy', 'Overlay', fy, 2000, on_distortion)
cv2.setTrackbarPos('fy', 'Overlay', fy)
cv2.createTrackbar('cx', 'Overlay', cx, rgb_width, on_distortion)
cv2.setTrackbarPos('cx', 'Overlay', cx)
cv2.createTrackbar('cy', 'Overlay', cy, rgb_height, on_distortion)
cv2.setTrackbarPos('cy', 'Overlay', cy)


# Mostrar la imagen inicial
update_overlay()

# Esperar eventos de teclas
while True:
    key = cv2.waitKey(0)
    on_key(key)
