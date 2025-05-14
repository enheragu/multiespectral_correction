#!/usr/bin/env python3
# encoding: utf-8

import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from utils.pickle_utils import load_pkl, save_pkl
from constants import output_data_path


c_blue = "#0171ba"
c_green = "#78b01c"
c_red = "#f23535" 

def to_homogeneous_2d(matrix_2d):
    homogeneous_2d = np.eye(3)
    homogeneous_2d[:2, :2] = matrix_2d[:2, :2]
    homogeneous_2d[:2, 2] = matrix_2d[:2, 2]
    return homogeneous_2d

def plot_camera_trajectories(trajectories, output_dir):
    for tag_name, (trajectory_visible, trajectory_lwir) in trajectories.items():
        fig, ax = plt.subplots(figsize=(24, 16))

        camera_visible_position = np.array([1, 1, 1])
        camera_lwir_position = np.array([1, 1, 1])

        camera_visible_trajectory = [camera_visible_position[:2]]
        camera_lwir_trajectory = [camera_lwir_position[:2]]

        for j, (transform_visible, transform_lwir) in enumerate(zip(trajectory_visible, trajectory_lwir)):
            homogeneous_transform_visible = to_homogeneous_2d(transform_visible)
            homogeneous_transform_lwir = to_homogeneous_2d(transform_lwir)
            camera_visible_position = np.dot(homogeneous_transform_visible, camera_visible_position)
            camera_lwir_position = np.dot(homogeneous_transform_lwir, camera_lwir_position)

            camera_visible_trajectory.append(camera_visible_position[:2])
            camera_lwir_trajectory.append(camera_lwir_position[:2])

        camera_visible_trajectory = np.array(camera_visible_trajectory)
        camera_lwir_trajectory = np.array(camera_lwir_trajectory)

        for visible_position, lwir_position in zip(camera_visible_trajectory[::5], camera_lwir_trajectory[::5]):
            ax.plot([visible_position[0], lwir_position[0]],
                    [visible_position[1], lwir_position[1]], '--', color=c_green, alpha=0.5)

        # Plotea las trayectorias completas
        ax.plot(camera_visible_trajectory[:, 0], camera_visible_trajectory[:, 1], '-', color=c_red, 
                label=f'Visible Camera ({len(camera_visible_trajectory)} points)')
        ax.plot(camera_lwir_trajectory[:, 0], camera_lwir_trajectory[:, 1], '-', color=c_blue, 
                label=f'LWIR Camera ({len(camera_lwir_trajectory)} points)')

        # Marca los puntos coincidentes
        ax.scatter(camera_visible_trajectory[::5, 0], camera_visible_trajectory[::5, 1], c=c_red, marker='o')
        ax.scatter(camera_lwir_trajectory[::5, 0], camera_lwir_trajectory[::5, 1], c=c_blue, marker='o')

            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Cameras trajectory - {tag_name}')
        ax.legend()
        ax.set_aspect('equal')

        plt.savefig(os.path.join(output_dir, f'trajectory_{tag_name}.png'), dpi=200)
        plt.close()



def listOpticalFlowFiles(carpeta):
    search_re = re.compile(r'^optical_flow_.*_.*\.pkl$')
    archivos_pkl = [f for f in os.listdir(carpeta) if search_re.match(f)]
    rutas_completas = [os.path.join(carpeta, f) for f in archivos_pkl]
    
    return rutas_completas

    
if __name__ == '__main__':
    output_directory = f'{output_data_path}/trajectory_projection'
    os.makedirs(output_directory, exist_ok=True)

    # Supongamos que tienes una lista de pares de trayectorias
    # Cada trayectoria es una lista de matrices de transformación afín 4x4
    output_directory_optical_flow = f'{output_data_path}/optical_flow'
    optical_flow_paths = listOpticalFlowFiles(output_directory_optical_flow)
    
    trajectories = {}
    for file in optical_flow_paths:
        data = load_pkl(file)

        search_re = re.compile(r'optical_flow_(set\d+_V\d{3})\.pkl')
        tag_name = search_re.search(file).group(1)

        visible_trajectory, lwir_trajectory = [], []
        for image_data in data:
            if image_data and image_data['oflow_visible']:
                visible_trajectory.append(np.array(image_data['oflow_visible']['transformation_matrix']))
            if image_data and image_data['oflow_lwir']:
                lwir_trajectory.append(np.array(image_data['oflow_lwir']['transformation_matrix']))
        trajectories[tag_name] = (visible_trajectory, lwir_trajectory)
    
    plot_camera_trajectories(trajectories, output_directory)
