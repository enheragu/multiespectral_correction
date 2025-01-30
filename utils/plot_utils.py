#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np

def estimate_distortion_center(origin_points, destination_points):
    vectors = destination_points - origin_points
    magnitudes = np.sqrt(np.sum(vectors**2, axis=1))
    magnitudes[magnitudes == 0] = 1e-10 # Avoid zero division
    weights = 1 / magnitudes
    weighted_coords = origin_points * weights[:, np.newaxis]
    center = np.sum(weighted_coords, axis=0) / np.sum(weights)
    return center


def plotDensityMap(image_points_lwir, image_points_rgb, removed1 = [], removed2 = [], image_size = [640,512], tag = ""):
    # PLOT DENSITY MAP OF POINTS
    width, height = image_size
    
    image_arrows = np.ones((height, width, 3), dtype=np.uint8) * 255
    image_points = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Plot distortion center and image center
    center = estimate_distortion_center(image_points_lwir, image_points_rgb)
    for image in [image_arrows,image_points]:
        cv2.circle(image, (int(center[0]), int(center[1])), 2, (0, 255, 0), -1)
        center_x, center_y = width // 2, height // 2
        cv2.line(image, (0, center_y), (width, center_y), (255, 255, 0), 1)
        cv2.line(image, (center_x, 0), (center_x, height), (255, 255, 0), 1)

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

