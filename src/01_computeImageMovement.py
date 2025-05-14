#!/usr/bin/env python3
# encoding: utf-8

"""
    Based on the optical flow it computes the relative transformation between two consecutive
    visible (RGB) images. The transformation is the one that transforms image Visible1 to
    Visible2 (being both consecutives in that respective order).
"""
import sys
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from collections import defaultdict
import argparse

from sklearn.linear_model import RANSACRegressor

from utils.pickle_utils import save_pkl, load_pkl
from constants import output_data_path, dataset_images_path


# def draw_flow(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    
#     # Filter out invalid indices
#     valid_y = y < flow.shape[0]
#     valid_x = x < flow.shape[1]
#     valid_indices = valid_y & valid_x
    
#     if np.any(valid_indices):
#         fx, fy = flow[y[valid_indices], x[valid_indices]].T
#         lines = np.vstack([x[valid_indices], y[valid_indices], 
#                            x[valid_indices]+fx, y[valid_indices]+fy]).T.reshape(-1, 2, 2)
#         lines = np.int32(lines + 0.5)
#         vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         cv2.polylines(vis, lines, 0, (255, 255, 0))
#         for (x1, y1), (_x2, _y2) in lines:
#             cv2.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
#     else:
#         tqdm.write(f"No valid indexes in flow shape")
#         tqdm.write(f"{flow.shape = }; {flow = }")
#         vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     return vis
def draw_flow(img, good_prev, good_curr):
    visialization = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for (prev, curr) in zip(good_prev, good_curr):
        x1, y1 = prev.ravel()
        x2, y2 = curr.ravel()
        
        cv2.line(visialization, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(visialization, (int(x2), int(y2)), 3, (0, 0, 255), -1)
    
    return visialization



class FrameMotionParams:
    def __init__(self, avg_movement = 0, rotation = 0, transformation_matrix = np.eye(3)[:2], translation_x = 0, translation_y = 0):
        self.avg_movement = avg_movement
        self.rotation = rotation
        self.transformation_matrix = transformation_matrix
        self.translation_x = translation_x
        self.translation_y = translation_y

    def to_dict(self):
        return {
            'avg_movement': self.avg_movement,
            'rotation': self.rotation,
            'transformation_matrix': self.transformation_matrix.tolist(),
            'translation_x': self.translation_x,
            'translation_y': self.translation_y
        }
  


def calculate_optical_flow(prev_frame, curr_frame, next_frame=None, debug=False, image_tag='visible'):
    # next_frame=None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    if curr_gray is None or prev_gray is None or curr_gray.size == 0 or prev_gray.size == 0:
        print(f"[ERROR] [calculate_optical_flow] Invalid input images")
        return FrameMotionParams()

    # Mejora el contraste
    curr_gray = cv2.equalizeHist(curr_gray)
    prev_gray = cv2.equalizeHist(prev_gray)

    # Check params of goodFeaturesToTrack
    curr_points = cv2.goodFeaturesToTrack(curr_gray, maxCorners=2000, qualityLevel=0.01, minDistance=10)
    if curr_points is None or not len(curr_points) > 0:
        print(f"[ERROR] [calculate_optical_flow] Could not get enough features to track in current frame!")
        return FrameMotionParams()

    curr_points = curr_points.astype(np.float32)

    # Check params from calcOpticalFlowPyrLK ?¿
    prev_points, status_prev, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_points, None, winSize=(21, 21), maxLevel=3)

    if next_frame is not None:
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        next_gray = cv2.equalizeHist(next_gray)
        next_points, status_next, _ = cv2.calcOpticalFlowPyrLK(curr_gray, next_gray, curr_points, None)
        good_points = (status_prev == 1) & (status_next == 1)
    else:
        good_points = (status_prev == 1)

    good_curr = curr_points[good_points].reshape(-1, 2)
    good_prev = prev_points[good_points].reshape(-1, 2)


    def computeAffine(curr_gray, prev_gray, good_curr, good_prev):
        # 1. Bidirectional optical flow (based on original data)
        prev_points_reverse, status_reverse, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, good_curr, None)
        good_reverse = (status_reverse == 1).reshape(-1)  # Asegurar que es 1D

        # 2. Transformación inicial (based on original data)
        M_prev_initial, _ = cv2.estimateAffinePartial2D(good_curr, good_prev, method=cv2.RANSAC)

        # 3. Distance threshold filter
        good_curr_proj = np.dot(good_curr, M_prev_initial[:, :2].T) + M_prev_initial[:, 2]
        distances = np.linalg.norm(good_prev - good_curr_proj, axis=1)

        median_distance = np.median(distances)
        std_distance = np.std(distances)
        threshold = median_distance + std_distance

        combined_mask = (distances < threshold) & good_reverse

        good_curr_filtered = good_curr[combined_mask]
        good_prev_filtered = good_prev[combined_mask]

        return cv2.estimateAffinePartial2D(good_curr_filtered, good_prev_filtered, method=cv2.RANSAC)

       
    if len(good_curr) > 4:
        M_prev, _ = computeAffine(curr_gray, prev_gray, good_curr, good_prev)

        if next_frame is not None:
            good_next = next_points[good_points].reshape(-1, 2)
            M_next, _ = computeAffine(next_gray, curr_gray, good_next, good_curr)

            weight_next = 0.1
            transformation_matrix = M_prev*(1-weight_next) + M_next*weight_next
        else:
            transformation_matrix = M_prev

        rotation = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
        translation_x = transformation_matrix[0, 2]
        translation_y = transformation_matrix[1, 2]

        flow = good_prev - good_curr
        avg_movement = np.mean(np.sqrt(np.sum(flow**2, axis=1)))

        if debug:
            global current_image_debug, current_corrected_image_debug, prev_image_debug
            flow_img = draw_flow(curr_gray, good_prev, good_curr) # Exagera el flujo visualmente
            cv2.imshow(f'Filtered Optical Flow {image_tag}', flow_img)

            current_image_debug = curr_frame.copy()
            h, w = curr_frame.shape[:2]
            current_corrected_image_debug = cv2.warpAffine(curr_frame.copy(), transformation_matrix, (w, h))
            prev_image_debug = prev_frame.copy()
            update_overlay()

            while True:
                key = cv2.waitKey(0)
                if key == 27 or  key == ord('q'):  # Tecla 'Esc' para salir
                    cv2.destroyAllWindows()
                    sys.exit("Program terminated by user")
                else:
                    break

        params = FrameMotionParams(avg_movement, rotation, transformation_matrix, translation_x, translation_y)
    else:
        params = FrameMotionParams()
        # Maybe use transformation from previous frame?¿

    return params

def process_sequence(args):
    sequence_data, index, debug = args
    set_name, sequence_name, pairs = sequence_data
    results = []    
    tqdm.write(f"Process images from: {set_name = }; {sequence_name = };")
    
    prev_frame_name = {'visible': None, 'lwir': None}
    curr_frame_name = {'visible': None, 'lwir': None}
    prev_frame = {'visible': None, 'lwir': None}
    curr_frame = {'visible': None, 'lwir': None}
    
    for i, (lwir_path, visible_path) in enumerate(tqdm(pairs, desc=f"{set_name}/{sequence_name}", position=index+1)):
        next_frame_name = {'visible': f"{dataset_images_path}/{visible_path}", 
                           'lwir': f"{dataset_images_path}/{lwir_path}"}
        next_frame = {'visible': cv2.imread(str(next_frame_name['visible'])), 
                      'lwir': cv2.imread(str(next_frame_name['lwir']))}
        
        if prev_frame['visible'] is not None and curr_frame['visible'] is not None:
            params_visible = calculate_optical_flow(prev_frame['visible'], curr_frame['visible'], next_frame['visible'], debug=debug)
            params_lwir = calculate_optical_flow(prev_frame['lwir'], curr_frame['lwir'], next_frame['lwir'], debug=debug)
            results.append({'lwir': lwir_path,
                            'visible': visible_path,
                            'oflow_visible': params_visible.to_dict(),
                            'oflow_lwir': params_lwir.to_dict()
                            })
            if debug:
                tqdm.write(f"Evaluating frames:\n\t· {prev_frame_name['visible'] = }\n\t· {curr_frame_name['visible'] = }\n\t· {next_frame_name['visible'] = }")
                tqdm.write(f"Params computed: {params_visible.to_dict()}")
            
        else:
            results.append({'lwir': lwir_path,
                            'visible': visible_path,
                            'oflow_visible': None,
                            'oflow_lwir': None
                            })
        prev_frame = curr_frame
        curr_frame = next_frame
        prev_frame_name = curr_frame_name
        curr_frame_name = next_frame_name
    
    # Process last frame
    if len(pairs) > 1 and prev_frame['visible'] is not None and curr_frame['visible'] is not None:
        params_visible = calculate_optical_flow(prev_frame['visible'], curr_frame['visible'], None, debug=debug)
        params_lwir = calculate_optical_flow(prev_frame['lwir'], curr_frame['lwir'], None, debug=debug)
        results.append({'lwir': pairs[-1][0],
                        'visible': pairs[-1][1],
                        'oflow_visible': params_visible.to_dict(),
                        'oflow_lwir': params_lwir.to_dict()
                        })
    
    output_pkl = f'{output_data_path}/optical_flow/optical_flow_{set_name}_{sequence_name}.pkl'
    tqdm.write(f"Saving results to {output_pkl}")
    save_pkl(results, output_pkl)
    return results, len(pairs)


current_image_debug = np.zeros((640, 512, 3))
current_corrected_image_debug = np.zeros((640, 512, 3))
prev_image_debug = np.zeros((640, 512, 3))
alpha = 0.5
def update_overlay():
    global alpha
    if current_image_debug is None or current_corrected_image_debug is None or prev_image_debug is None:
        tqdm.write("[Error] [update_overlay] One or more images are not loaded.")
        return
    
    overlay = cv2.addWeighted(prev_image_debug, 1 - alpha, current_image_debug, alpha, 0)
    cv2.imshow(f'Prev {debug_spectr} + Current {debug_spectr}', overlay)

    overlay_corrected = cv2.addWeighted(prev_image_debug, 1 - alpha, current_corrected_image_debug, alpha, 0)
    cv2.imshow(f'Prev {debug_spectr} + Current Projected Corrected', overlay_corrected)

    def getDiff(prev_image_debug, curr_image):
        if prev_image_debug.dtype != np.uint8:
            prev_image_debug = (prev_image_debug * 255).astype(np.uint8)
        if curr_image.dtype != np.uint8:
            curr_image = (curr_image * 255).astype(np.uint8)

        diff = cv2.absdiff(prev_image_debug, curr_image)
        if len(diff.shape) == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        norm_diff = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX)
        
        return norm_diff
    
    def get_correction_factor(diff_original, diff_corrected):
        if diff_original.dtype != np.uint8:
            diff_original = (diff_original * 255).astype(np.uint8)
        if diff_corrected.dtype != np.uint8:
            diff_corrected = (diff_corrected * 255).astype(np.uint8)

        white_area_original = np.sum(diff_original == 255)
        white_area_corrected = np.sum(diff_corrected == 255)

        if white_area_original > 0:
            correction_factor = (white_area_original - white_area_corrected) / white_area_original
        else:
            correction_factor = 0

        return correction_factor

    diff_original = getDiff(prev_image_debug, current_image_debug)
    diff_corrected = getDiff(prev_image_debug, current_corrected_image_debug)

    cv2.imshow('Diff original', diff_original)
    cv2.imshow('Diff corrected', diff_corrected)
    tqdm.write(f"Correction factor of {get_correction_factor(diff_original,diff_corrected)}")



def on_trackbar(val):
    global alpha
    alpha = val / 100
    update_overlay()
    cv2.setTrackbarPos(f'Transparency', f'Prev {debug_spectr} + Current {debug_spectr}', val)
    cv2.setTrackbarPos(f'Transparency', f'Prev {debug_spectr} + Current Projected Corrected', val)


debug_spectr = 'lwir'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and calculate optical flow.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode to show optical flow visualization')
    args = parser.parse_args()

    if args.debug:
        cv2.namedWindow(f'Prev {debug_spectr} + Current {debug_spectr}')
        cv2.namedWindow(f'Prev {debug_spectr} + Current Projected Corrected')
        cv2.createTrackbar(f'Transparency', f'Prev {debug_spectr} + Current {debug_spectr}', 50, 100, on_trackbar)
        cv2.createTrackbar(f'Transparency', f'Prev {debug_spectr} + Current Projected Corrected', 50, 100, on_trackbar)
        update_overlay()

    pkl_path = f'{output_data_path}/image_pairs.pkl'

    tqdm.write(f"Load image pair data from {pkl_path}")
    image_pairs = load_pkl(pkl_path)
    
    sorted_pairs = defaultdict(list)
    for pair in image_pairs:
        lwir_path, visible_path = pair
        set_name = Path(visible_path).parts[1]
        sequence_name = Path(visible_path).parts[2]
        sorted_pairs[(set_name, sequence_name)].append((lwir_path, visible_path))
    
    for key in sorted_pairs:
        sorted_pairs[key].sort(key=lambda x: x[1])  # x[1] es 

    sequences_to_process = [(set_name, sequence_name, pairs) for (set_name, sequence_name), pairs in sorted_pairs.items()]
    
    tqdm.write(f"Start processing parallel sequences.")
    total_pairs = sum(len(pairs) for _, _, pairs in sequences_to_process)

    if args.debug:
        for index, seq_data in enumerate(sequences_to_process):
            process_sequence((seq_data, index, args.debug)) 
    else:
        max_workers = 8 # multiprocessing.cpu_count()
        with tqdm(total=total_pairs, desc="Overall Progress", position=0) as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_sequence, (seq_data, index, args.debug)) for index, seq_data in enumerate(sequences_to_process)]
                
                for future in as_completed(futures):
                    results, processed_pairs = future.result()
                    pbar.update(processed_pairs)

    print("Finished processing all data")