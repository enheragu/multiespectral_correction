#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from tqdm import tqdm
import os

from utils.get_data import getPointData
from utils.filter_distortionvector_grid import filterVectorsWithCalibration
from utils.affine_transform_utils import scaleAffineTransform, applyTransformation
from utils.load_optical_flow import load_optical_flow
from utils.pickle_utils import load_pkl, save_pkl
from constants import output_data_path


def filterOutliers(data, m=1.0):
    """ Filtra valores que se desvían más de `m` desviaciones estándar. """
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        return data
    mean = np.mean(data)
    std_dev = np.std(data)
    return data[np.abs(data - mean) < m * std_dev] if std_dev > 0 else data


if __name__ == '__main__':
    print(f"Load labeled correspondences between LWIR-Visible pairs")
    data_path = f"{output_data_path}/he_points.pkl"
    points_data = load_pkl(data_path)

    # Filtrar puntos incoherentes
    image_points_lwir, image_points_rgb = getPointData(data_path)
    filtered_data = filterVectorsWithCalibration(image_points_lwir, image_points_rgb)
    filtered_image_points_lwir, filtered_image_points_rgb, removed_lwir, removed_rgb = filtered_data

    removed_lwir_set = {tuple(pt) for pt in removed_lwir}
    removed_rgb_set = {tuple(pt) for pt in removed_rgb}

    fractions_by_subset = {}
    total_points = sum(len(data['lwir']) for data in points_data.values())

    translation_x_list = []
    translation_y_list = []

    with tqdm(total=total_points, desc="Processing points") as pbar:
        for lwir_file, data in points_data.items():
            lwir_points = data['lwir']
            rgb_points = data['rgb']
            rgb_file = data['rgb_file']

            path_parts = rgb_file.split("/")
            if len(path_parts) < 3:
                tqdm.write(f"[Warning] Unexpected path format: {rgb_file}")
                continue

            subset_key = f"{path_parts[1]}/{path_parts[2]}"  # "set01/V002"

            flow_data = load_optical_flow(rgb_file)
            visible_flow = flow_data.get('oflow_visible', None)

            if visible_flow is None:
                tqdm.write(f"[Warning] No optical flow data for {rgb_file}")
                continue

            translation_x_list.append(visible_flow["translation_x"])
            translation_y_list.append(visible_flow["translation_y"])
            transform_matrix = np.array(visible_flow['transformation_matrix'])

            valid_count = 0
            for lwir_point, rgb_point in zip(lwir_points, rgb_points):
                lwir_tuple = tuple(lwir_point)
                rgb_tuple = tuple(rgb_point)

                # Filtrar solo si el punto está en ambas listas de eliminados
                if lwir_tuple in removed_lwir_set and rgb_tuple in removed_rgb_set:
                    continue
                
                lwir_point = np.array(lwir_point)
                rgb_point = np.array(rgb_point)

                prev_rgb_point = applyTransformation(rgb_point, transform_matrix)
                lwir_rgb_displacement = np.linalg.norm(lwir_point - rgb_point)
                rgb_pairs_displacement = np.linalg.norm(prev_rgb_point - rgb_point)

                if rgb_pairs_displacement > 0:
                    fraction = lwir_rgb_displacement / rgb_pairs_displacement
                    fractions_by_subset.setdefault(subset_key, []).append(fraction)

                valid_count += 1

            pbar.update(valid_count)

    print(f"Calibration filtering removed {len(removed_lwir)} points. {len(filtered_image_points_lwir)} point pairs left.")

    all_fractions = []
    distortion_fraction = {}
    print("\n### Distortion fraction per subset ###")
    for subset, fractions in fractions_by_subset.items():
        filtered_fractions = filterOutliers(np.array(fractions))
        if len(filtered_fractions) > 0:
            subset_n = len(filtered_fractions)
            subset_mean = np.mean(filtered_fractions)
            subset_std = np.std(filtered_fractions)
            print(f"\t· {subset}: (n={subset_n}) Avg = {subset_mean:.4f}, Std = {subset_std:.4f}")
            all_fractions.extend(filtered_fractions)
            distortion_fraction[subset] = {'n': subset_n, 'mean': subset_mean, 'std': subset_std}
        else:
            print(f"{subset}: No valid fractions after filtering.")

    if all_fractions:
        filtered_all = filterOutliers(np.array(all_fractions))
        if len(filtered_all) > 0:
            all_n = len(filtered_all)
            all_mean = np.mean(filtered_all)
            all_std = np.std(filtered_all)
            print(f"\n### Overall Distortion Fraction ###")
            print(f"\t· all: (n={all_n}) Avg = {all_mean:.4f}, Std = {all_std:.4f}")
            distortion_fraction["all"] = {'n': all_n, 'mean': all_mean, 'std': all_std}
        else:
            print("\n[Warning] No valid data after filtering!")
    else:
        print("\n[Warning] No valid fractions were collected!")

    trans_x_mean = np.mean(translation_x_list)
    trans_x_std = np.std(translation_x_list)
    trans_y_mean = np.mean(translation_y_list)
    trans_y_std = np.std(translation_y_list)

    print(f"\n### Translation Stats ###")
    print(f"Translation X -> mean: {trans_x_mean:.4f}, std: {trans_x_std:.4f}")
    print(f"Translation Y -> mean: {trans_y_mean:.4f}, std: {trans_y_std:.4f}")
    print(f"Translation X limits = [{trans_x_mean - 2 * trans_x_std}, {trans_x_mean + 2 * trans_x_std}]")
    print(f"Translation Y limits = [{trans_y_mean - 2 * trans_y_std}, {trans_y_mean + 2 * trans_y_std}]")

    # Escalar la transformación usando la fracción promedio general
    if "all" in distortion_fraction:
        scaled_transform = scaleAffineTransform(transform_matrix, distortion_fraction["all"]["mean"])
        print("\nScaled transformation matrix:")
        print(scaled_transform)

    output_pkl = f'{output_data_path}/transform_fraction.pkl'
    tqdm.write(f"Saving results to {output_pkl}")
    save_pkl(distortion_fraction, output_pkl)
