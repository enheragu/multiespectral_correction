#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.affine_transform_utils import scaleAffineTransform, invertAffineTransform
from utils.load_optical_flow import load_optical_flow
from utils.pickle_utils import load_pkl, save_pkl
from constants import output_data_path


## Kaist is configured the following way:
# Most of labels at night are associated to the LWIR image, needs to be taken into
# account when transforming data.
# During daylight LWIR imag ewill be corrected to match RGB image, during night
# the other way around.
### Train ###
# · Set 00 - Day / Campus / 5.92GB / 17,498 frames / 11,016 objects
# · Set 01 - Day / Road / 2.82GB / 8,035 frames / 8,550 objects
# · Set 02 - Day / Downtown / 3.08GB / 7,866 frames / 11,493 objects
# · Set 03 - Night / Campus / 2.40GB / 6,668 frames / 7,418 objects
# · Set 04 - Night / Road / 2.88GB / 7,200 frames / 17,579 objects
# · Set 05 - Night / Downtown / 1.01GB / 2,920 frames / 4,655 objects
    
### Test ###
# · Set 06 - Day / Campus / 4.78GB / 12,988 frames / 12,086 objects
# · Set 07 - Day / Road / 3.04GB / 8,141 frames / 4,225 objects
# · Set 08 - Day / Downtown / 3.50GB / 8,050 frames / 23,309 objects
# · Set 09 - Night / Campus / 1.38GB / 3,500 frames / 3,577 objects
# · Set 10 - Night / Road / 3.75GB / 8,902 frames / 4,987 objects
# · Set 11 - Night / Downtown / 1.33GB / 3,560 frames / 6,655 objects

day_sets = ('set00','set01','set02','set06','set07','set08')
night_sets = ('set03','set04','set05','set09','set10','set11')

if __name__ == '__main__':
    pkl_path = f'{output_data_path}/image_pairs.pkl'
    tqdm.write(f"Load image pair data from {pkl_path}")
    image_pairs = load_pkl(pkl_path)

    tranform_data = {}
    with tqdm(total=len(image_pairs), desc="Processing points") as pbar:
        for lwir_image, rgb_image in image_pairs:
            set_name = Path(rgb_image).parts[1]
            sequence_name = Path(rgb_image).parts[2]
            visible_flow = load_optical_flow(rgb_image)

            if not visible_flow:
                tqdm.write(f"[ERROR] Processing data from {lwir_image, rgb_image} images")
                continue

            if not visible_flow['oflow_visible']:
                tqdm.write(f"[ERROR] Processing data from {visible_flow}")
                continue
            
            if not (set_name,sequence_name) in tranform_data:
                tranform_data[(set_name,sequence_name)] = {}

            visible2visible_transform = np.array(visible_flow['oflow_visible']['transformation_matrix'])
            visible2lwir_transform = scaleAffineTransform(invertAffineTransform(visible2visible_transform),5)
           
            if set_name in day_sets:
                lwir2visible_transform = invertAffineTransform(visible2lwir_transform)
                tranform_data[(set_name,sequence_name)][lwir_image] = lwir2visible_transform
            elif set_name in night_sets:
                tranform_data[(set_name,sequence_name)][rgb_image] = visible2lwir_transform
            else:
                tqdm.write(f"[ERROR] {set_name} does not corresponds to any set for day/night conditions")    

            pbar.update(1)
    
    for key, items in tranform_data.items():
        output_pkl = f'{output_data_path}/transform_{key[0]}_{key[1]}.pkl'
        tqdm.write(f"Saving results to {output_pkl}")
        save_pkl(items, output_pkl)

