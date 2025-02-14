

import yaml
import numpy as np

class NumpyLoader(yaml.SafeLoader):
    pass

def numpy_constructor(loader, node):
    if isinstance(node, yaml.ScalarNode):
        return np.array(loader.construct_scalar(node))
    elif isinstance(node, yaml.SequenceNode):
        return np.array(loader.construct_sequence(node))
    elif isinstance(node, yaml.MappingNode):
        return np.array(loader.construct_mapping(node))

NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', numpy_constructor)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=NumpyLoader)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
