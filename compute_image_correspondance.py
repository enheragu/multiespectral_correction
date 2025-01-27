#!/usr/bin/env python3
# encoding: utf-8

import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
import scipy.interpolate as interp

IMAGE_SIZE = 640,512

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


def calibrateImagePair(image_points_lwir, image_points_rgb, image_size = IMAGE_SIZE):
    # Assumes RGB camera points are the correct ones
    object_points = np.zeros((len(image_points_lwir), 3), dtype=np.float32)
    object_points[:, :2] = image_points_rgb  # Los puntos 2D de la cámara RGB

    # Matriz de la cámara (estimación inicial, f y c son aproximados)
    fx, fy = 765, 765
    cx, cy = image_size[0] // 2, image_size[1] // 2

    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Coeficientes de distorsión radial y tangencial init to 0

    ret, camera_matrix_est, dist_coeffs_est, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [image_points_lwir], (640, 360), camera_matrix, dist_coeffs
    )

    print(f"Matriz de la cámara estimada with {len(image_points_lwir)} points:")
    print(camera_matrix_est)
    print(f"Coeficientes de distorsión estimados with {len(image_points_lwir)} points:")
    print(dist_coeffs_est)

    return camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points

# Bin size in pixels
def computeReprojectionError(camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points, image_points_lwir, store = None, bin_size = 5):
    image_points_proj, _ = cv2.projectPoints(object_points, rvecs[0], tvecs[0], camera_matrix_est, dist_coeffs_est)
    error_list = np.sqrt(np.sum((image_points_lwir - image_points_proj.reshape(-1, 2))**2, axis=1))
    mean_error = np.mean(error_list)

    print(f"Error de reproyección medio: {mean_error} píxeles")
    
    range_error = np.max(error_list) - np.min(error_list)
    num_bins = int(np.ceil(range_error / bin_size))
    print(f"Number of bins: {num_bins}")
    print(f"Size of bin: {bin_size:.2f} pixel")

    plt.figure(figsize=(8, 6))
    plt.hist(error_list, bins=num_bins, color='blue', edgecolor='black')
    plt.title('Reprojection error histogram')
    plt.xlabel('Repojection Error (pixel)')
    plt.ylabel('Freq.')
    plt.grid(True)
    if store is not None:
        plt.savefig(store)

    return mean_error, error_list

"""
    Computes adaptative thresholds based on neighbours for each point

    :neighborhood_radius: in pixels how much distance does it take into account to average errors
    :factor: multiplier for max threshold (error*factor)
"""
def compute_adaptive_thresholds(points, errors, neighborhood_radius, factor=2):
    tree = cKDTree(points)
    thresholds = np.zeros_like(errors)

    for i, point in enumerate(points):
        neighbors_idx = tree.query_ball_point(point, r=neighborhood_radius)
        local_errors = errors[neighbors_idx]
        thresholds[i] = np.mean(local_errors) * factor
    return thresholds

def filterPointsWithError(image_points_lwir, image_points_rgb, error_list):
    neighborhood_radius = 25 # in pixels
    adaptive_factor = 2
    adaptive_thresholds = compute_adaptive_thresholds(image_points_lwir, error_list, neighborhood_radius, adaptive_factor)

    filtered_image_points_lwir = []
    filtered_image_points_rgb = []
    removed1 = []
    removed2 = []

    for i, (point, error) in enumerate(zip(image_points_lwir, error_list)):
        if error <= adaptive_thresholds[i]:
            filtered_image_points_lwir.append(image_points_lwir[i])
            filtered_image_points_rgb.append(image_points_rgb[i])
        else:
            removed1.append(image_points_lwir[i])
            removed2.append(image_points_rgb[i])

    filtered_image_points_lwir = np.array(filtered_image_points_lwir, dtype=np.float32)
    filtered_image_points_rgb = np.array(filtered_image_points_rgb, dtype=np.float32)
    
    print(f"Due to max error threshold (adaptative with mean of {np.mean(adaptive_thresholds)} pixels) {len(removed1)} points are not taken into account.")
    return filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2



def generateCorrectedImage(camera_matrix_est, dist_coeffs_est,rvecs,tvecs,tag = '_corrected'):
    ## Get corrected images!
    imagen_lwir_path = "images/lwir_I01689.png"
    imagen_rgb_path = "images/visible_I01689.png"
    imagen_lwir = cv2.imread(imagen_lwir_path)
    imagen_rgb = cv2.imread(imagen_rgb_path)
    # imagen_lwir_corrected = cv2.undistort(imagen_lwir, camera_matrix_est, dist_coeffs_est)
    R, _ = cv2.Rodrigues(rvecs[0])  # rvecs[0] viene de la calibración
    T = tvecs[0]  # tvecs[0] también viene de la calibración
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix_est, dist_coeffs_est, (imagen_lwir.shape[1], imagen_lwir.shape[0]), alpha=0
    ) # New projection matrix

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix_est, dist_coeffs_est, None, new_camera_matrix,
        (imagen_lwir.shape[1], imagen_lwir.shape[0]), cv2.CV_32FC1
    )
    imagen_lwir_corrected = cv2.remap(imagen_lwir, map1, map2, interpolation=cv2.INTER_LINEAR)

    x, y, w, h = roi
    imagen_lwir_cropped = imagen_lwir_corrected[y:y+h, x:x+w]
    imagen_rgb_cropped = imagen_rgb[y:y+h, x:x+w]

    # cv2.imshow('LWIR Image Corrected', imagen_lwir_corrected)
    os.makedirs('plot', exist_ok=True)
    cv2.imwrite(f'plot/{imagen_lwir_path.replace(".png", f"{tag}.png")}', imagen_lwir_cropped)
    cv2.imwrite(f'plot/{imagen_rgb_path.replace(".png", f"{tag}.png")}', imagen_rgb_cropped)




def plotDensityMap(image_points_lwir, image_points_rgb, removed1 = [], removed2 = [], image_size = IMAGE_SIZE, tag = ""):
    # PLOT DENSITY MAP OF POINTSimage_size = (640, 512)  # (ancho, alto)
    width, height = image_size
    points = image_points_rgb  # Usaremos los puntos RGB para generar el mapa de densidad

    image_arrows = np.ones((height, width, 3), dtype=np.uint8) * 255
    image_points = np.ones((height, width, 3), dtype=np.uint8) * 255

    for point1, point2 in zip(image_points_lwir, image_points_rgb):
        x, y = point2
        if 0 <= x < width and 0 <= y < height: 
            cv2.circle(image_points, (int(x), int(y)), 2, (255, 0, 0), -1)
            cv2.arrowedLine(image_arrows, (int(point1[0]),int(point1[1])), (int(point2[0]),int(point2[1])), (255, 0, 0), 1, cv2.LINE_AA, tipLength=0.2)


    for point1, point2 in zip(removed1,removed2):
        x, y = point2
        if 0 <= x < width and 0 <= y < height: 
            cv2.circle(image_points, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.arrowedLine(image_arrows, (int(point1[0]),int(point1[1])), (int(point2[0]),int(point2[1])), (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.2)
            

    cv2.putText(image_arrows, f"Used {len(image_points_rgb)}; rejected {len(removed1)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(image_points, f"Used {len(image_points_rgb)}; rejected {len(removed1)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(f'plot/calib_arrows{tag}.png', image_arrows)
    cv2.imwrite(f'plot/calib_points{tag}.png', image_points)



# Based on vectors (lwir to rgb points) average them in areas and interpolate to cover the whole image
def filterInterpolateVectors(image_points_lwir, image_points_rgb, image_size, grid_npoints = 40):
    vectors = image_points_rgb - image_points_lwir
    vector_lengths = np.linalg.norm(vectors, axis=1)

    # grid_x, grid_y = np.mgrid[0:image_size[0]:complex(0, grid_npoints), 0:image_size[1]:complex(0, grid_npoints)]  # 50 points for each axis
    grid_x, grid_y = np.meshgrid( # meshgrid is uniform in space
        np.linspace(0, image_size[0], grid_npoints),
        np.linspace(0, image_size[1], grid_npoints)
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Search for each cell all points in the selected area
    tree = cKDTree(image_points_lwir)
    cell_radius = int((image_size[0]) / grid_npoints)*4

    vector_field_x = np.zeros(grid_points.shape[0])
    vector_field_y = np.zeros(grid_points.shape[0])
    for i, grid_point in enumerate(grid_points):
        # Encuentra los puntos dentro del radio de la celda
        indices = tree.query_ball_point(grid_point, cell_radius)
        if indices:            
            # Calcular el promedio de los vectores en ese área
            mean_vector = np.average(vectors[indices], axis=0)
            vector_field_x[i] = mean_vector[0]
            vector_field_y[i] = mean_vector[1]

    interpolated_lwir = grid_points
    interpolated_rgb = interpolated_lwir + np.column_stack(
        [vector_field_x.ravel(), vector_field_y.ravel()]
    )
    plotDensityMap(interpolated_lwir, interpolated_rgb, image_size=IMAGE_SIZE, tag="_averagefilter")

    # Reshapear el campo vectorial para ajustarlo a la cuadrícula
    vector_field_x = vector_field_x.reshape(grid_x.shape)
    vector_field_y = vector_field_y.reshape(grid_y.shape)

    # Averaged
    vector_field_x = interp.griddata(image_points_lwir, vectors[:, 0], (grid_x, grid_y), method='linear')
    vector_field_y = interp.griddata(image_points_lwir, vectors[:, 1], (grid_x, grid_y), method='linear')

    # plt.figure(figsize=(10, 8))
    # plt.quiver(grid_x, grid_y, vector_field_x, vector_field_y, color="blue", alpha=0.6, scale=50, width=0.003)
    # # plt.scatter(image_points_lwir[:, 0], image_points_lwir[:, 1], color="red", label="Puntos de origen")
    # # plt.scatter(image_points_rgb[:, 0], image_points_rgb[:, 1], color="green", label="Puntos de destino")
    # plt.title("Campo de vectores promedio e interpolado")
    # plt.legend()
    # plt.show()

    interpolated_lwir = grid_points
    interpolated_rgb = interpolated_lwir + np.column_stack(
        [vector_field_x.ravel(), vector_field_y.ravel()]
    )
    return interpolated_lwir, interpolated_rgb


## GET POINTS FROM YAML FILE
image_points_lwir, image_points_rgb = getPointData()

## CALIBRATE PAIR AND FILTER DATA BASED ON REPROJECTION ERROR
calibrate_data = calibrateImagePair(image_points_lwir, image_points_rgb)
mean_reprojection_error, error_list = computeReprojectionError(*calibrate_data, image_points_lwir, 'plot/histogram_reprojection.png')
filtered_data = filterPointsWithError(image_points_lwir, image_points_rgb, error_list)
filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2 = filtered_data

## CALIBRATE AGAIN BASED ON FILTERED DATA AND LOG
calibrate_data = calibrateImagePair(filtered_image_points_lwir, filtered_image_points_rgb)
camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points = calibrate_data
mean_reprojection_error, error_list = computeReprojectionError(*calibrate_data, filtered_image_points_lwir, 'plot/histogram_reprojection_filtered.png', bin_size=2)
generateCorrectedImage(camera_matrix_est, dist_coeffs_est,rvecs,tvecs)
plotDensityMap(filtered_image_points_lwir, filtered_image_points_rgb, removed1, removed2, IMAGE_SIZE, tag="_wremoved")
plotDensityMap(filtered_image_points_lwir, filtered_image_points_rgb, IMAGE_SIZE)

## FILTER AND INTERPOLATE DATA BASED ON DISTORTION VECTOR FOR EACH POINT AND CALIBRATE AGAIN
interpolated_lwir, interpolated_rgb = filterInterpolateVectors(filtered_image_points_lwir, filtered_image_points_rgb, IMAGE_SIZE)
plotDensityMap(interpolated_lwir, interpolated_rgb, image_size=IMAGE_SIZE, tag="_gridsampled")
calibrate_data = calibrateImagePair(filtered_image_points_lwir, filtered_image_points_rgb)
camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points = calibrate_data
mean_reprojection_error, error_list = computeReprojectionError(*calibrate_data, filtered_image_points_lwir, 'plot/histogram_reprojection_filtered.png', bin_size=2)
generateCorrectedImage(camera_matrix_est, dist_coeffs_est,rvecs,tvecs)

cv2.pollKey()
# plt.show()
cv2.destroyAllWindows()