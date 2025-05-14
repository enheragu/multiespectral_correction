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

# sets_scale_factor = {'set00': 2.5,'set01': 5,'set02': 5,'set06': 5,'set07': 5,'set08': 5, # dat
#           'set03': 1,'set04': 2,'set05': 2,'set09': 2,'set10': 2,'set11': 2} # night
sets_scale_factor = { # Manually adjusted and reviewed from 02_computeMultiespectralDistortionFactor.py -> very big numbers might appear do to labelled moving parts in the image (coupled car movement+moving object)
                    'set00/V000': 1.6, # Day sets
                    'set00/V001': 1.8,
                    'set00/V002': 3.0,
                    'set00/V003': 5.0,
                    'set00/V004': 1.4,
                    'set00/V005': 1.8,
                    'set00/V006': 1.9,
                    'set00/V007': 1.7,
                    'set00/V008': 1.7,
                    'set01/V000': 4,
                    'set01/V001': 4.9,
                    'set01/V002': 2.0,
                    'set01/V003': 1.1,
                    'set01/V004': 3,
                    'set01/V005': 3,
                    'set02/V000': 3.6,
                    'set02/V001': 2.8,
                    'set02/V002': 3,
                    'set02/V003': 2.7,
                    'set02/V004': 2.2, 
                    'set03/V000': 2.2, # Night sets
                    'set03/V001': 2.8,
                    'set04/V000': 3,
                    'set04/V001': 3,
                    'set05/V000': 3.1,
                    'set06/V000': 1.2, # Day sets
                    'set06/V001': 2.0,
                    'set06/V002': 1.8,
                    'set06/V003': 2.3,
                    'set06/V004': 2.1,
                    'set07/V000': 1.6,
                    'set07/V001': 3.3,
                    'set07/V002': 2.9,
                    'set08/V000': 2.2,
                    'set08/V001': 3,
                    'set08/V002': 6.0,
                    'set09/V000': 3.0, # Night sets
                    'set10/V000': 2,
                    'set10/V001': 2,
                    'set11/V000': 2.5,
                    'set11/V001': 4.0,
                    }


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
            # if set_name in list(day_sets_t.keys()):
            visible2lwir_transform = scaleAffineTransform(invertAffineTransform(visible2visible_transform), sets_scale_factor[f"{set_name}/{sequence_name}"])
            lwir2visible_transform = invertAffineTransform(visible2lwir_transform)
            tranform_data[(set_name,sequence_name)][lwir_image] = lwir2visible_transform
            # elif set_name in list(night_sets_t.keys()):
            #     visible2lwir_transform = scaleAffineTransform(invertAffineTransform(visible2visible_transform),night_sets_t[set_name]/2)
            #     lwir2visible_transform = invertAffineTransform(visible2lwir_transform)
            #     tranform_data[(set_name,sequence_name)][rgb_image] = visible2lwir_transform
            #     tranform_data[(set_name,sequence_name)][lwir_image] = lwir2visible_transform

            pbar.update(1)
    
    for key, items in tranform_data.items():
        output_pkl = f'{output_data_path}/transform/transform_{key[0]}_{key[1]}.pkl'
        tqdm.write(f"Saving results to {output_pkl}")
        save_pkl(items, output_pkl)

