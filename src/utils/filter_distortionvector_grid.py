#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

IMAGE_SIZE = 640,512


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
def computeReprojectionError(camera_matrix_est, dist_coeffs_est, rvecs, tvecs, object_points, image_points_lwir, store = None, bin_size = 5, plot = False):
    image_points_proj, _ = cv2.projectPoints(object_points, rvecs[0], tvecs[0], camera_matrix_est, dist_coeffs_est)
    error_list = np.sqrt(np.sum((image_points_lwir - image_points_proj.reshape(-1, 2))**2, axis=1))
    mean_error = np.mean(error_list)

    print(f"Error de reproyección medio: {mean_error} píxeles")
    
    range_error = np.max(error_list) - np.min(error_list)
    num_bins = int(np.ceil(range_error / bin_size))
    print(f"Number of bins: {num_bins}")
    print(f"Size of bin: {bin_size:.2f} pixel")

    if plot:
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
    adaptive_factor = 2.8
    adaptive_thresholds = compute_adaptive_thresholds(image_points_lwir, error_list, neighborhood_radius, adaptive_factor)

    filtered_image_points_lwir = []
    filtered_image_points_rgb = []
    removed_lwir = []
    removed_rgb = []

    for i, (point, error) in enumerate(zip(image_points_lwir, error_list)):
        if error <= adaptive_thresholds[i]:
            filtered_image_points_lwir.append(image_points_lwir[i])
            filtered_image_points_rgb.append(image_points_rgb[i])
        else:
            removed_lwir.append(image_points_lwir[i])
            removed_rgb.append(image_points_rgb[i])

    filtered_image_points_lwir = np.array(filtered_image_points_lwir, dtype=np.float32)
    filtered_image_points_rgb = np.array(filtered_image_points_rgb, dtype=np.float32)
    
    print(f"Due to max error threshold (adaptative with mean of {np.mean(adaptive_thresholds)} pixels) {len(removed_lwir)} points are not taken into account.")
    return filtered_image_points_lwir, filtered_image_points_rgb, removed_lwir, removed_rgb

def filterVectorsWithCalibration(image_points_lwir, image_points_rgb):
    calibrate_data = calibrateImagePair(image_points_lwir, image_points_rgb)
    mean_reprojection_error, error_list = computeReprojectionError(*calibrate_data, image_points_lwir, 'plot/histogram_reprojection.png')
    filtered_data = filterPointsWithError(image_points_lwir, image_points_rgb, error_list)
    return filtered_data


"""
    Applies a weighted average filter to the vectors

    Params:
        image_points_lwir (np.array): Set of points (Nx2 or Nx3) in the LWIR image.
        image_points_rgb (np.array): Corresponding set of points in the RGB image.
        N (int): Number of neighbors to consider for filtering.
        sigma (float): Parameter for Gaussian weighting.
    Returns:
        np.array, np.array: Filtered versions of image_points_lwir and image_points_rgb.
"""
def filterVectorsWithNeighbours(image_points_lwir, image_points_rgb, N=8, sigma=1.0):
    def weighted_average(values, distances):
        weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))
        weights[weights == 0] = 1e-10 # Avoid zero division
        return np.average(values, axis=0, weights=weights)
    
    def compute_filtered_vectors(origins, destinations):
        vectors = destinations - origins
        filtered_vectors = np.zeros_like(vectors)
        
        print(f"Compute filtered for {len(vectors)} vectors")
        for i, v in enumerate(vectors):
            distances = np.linalg.norm(vectors - v, axis=1)
            nearest_indices = np.argsort(distances)[1:N+1]  # Exclude the vector itself
            nearest_vectors = vectors[nearest_indices]
            nearest_distances = distances[nearest_indices]
            
            avg_vector = weighted_average(nearest_vectors, nearest_distances)
            filtered_vectors[i] = avg_vector
        
        filtered_destinations = origins + filtered_vectors
        return origins, filtered_destinations
    
    filtered_lwir, filtered_rgb = compute_filtered_vectors(image_points_lwir, image_points_rgb)
    
    return filtered_lwir, filtered_rgb