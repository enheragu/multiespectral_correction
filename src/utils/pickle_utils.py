

import pickle

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_pkl(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def yaml2pkl(file_path):
    from utils.yaml_utils import load_yaml
    data = load_yaml(file_path)
    pkl_path = file_path.replace('.yaml', '.pkl')
    save_pkl(data, pkl_path)
    