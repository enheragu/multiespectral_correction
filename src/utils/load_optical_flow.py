

from pathlib import Path
from utils.pickle_utils import load_pkl

from constants import output_data_path

optical_flow_cache = {}
def load_optical_flow(rgb_file_path, optical_flow_data_path = output_data_path):
    global optical_flow_cache

    set_name = Path(rgb_file_path).parts[1]
    sequence_name = Path(rgb_file_path).parts[2]
    if f"{set_name}/{sequence_name}" not in optical_flow_cache:
        optical_flow_file = f'{optical_flow_data_path}/optical_flow_{set_name}_{sequence_name}.pkl'
        optical_flow_cache[f"{set_name}/{sequence_name}"] = load_pkl(optical_flow_file)
    optical_flow_list = optical_flow_cache[f"{set_name}/{sequence_name}"]
    try:
        flow_data = next(item for item in optical_flow_list if item['visible'] == rgb_file_path)
    except StopIteration as e:
        print(f"Could not get {rgb_file_path} item in current dictionary from {set_name}/{sequence_name} cache. Current available items are: {[item['visible'] for item in optical_flow_list]}")
    return flow_data

