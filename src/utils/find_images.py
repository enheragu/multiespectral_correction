

from glob import glob

from utils.pickle_utils import save_pkl
from constants import output_data_path

# Función para encontrar pares de imágenes (lwir y visible)
def find_image_pairs(base_path):
    print(f"[INFO] Buscando imágenes en {base_path}...")
    lwir_paths = glob(f"{base_path}/**/lwir/*.png", recursive=True) + glob(f"{base_path}/**/lwir/*.jpg", recursive=True)
    visible_paths = glob(f"{base_path}/**/visible/*.png", recursive=True) + glob(f"{base_path}/**/visible/*.jpg", recursive=True)

    # Crear un diccionario de pares por nombre de archivo
    lwir_dict = {p.replace(base_path, '').replace('lwir/', ''): p.replace(base_path, '') for p in lwir_paths}
    visible_dict = {p.replace(base_path, '').replace('visible/', ''): p.replace(base_path, '') for p in visible_paths}

    # Encontrar nombres comunes
    common_names = set(lwir_dict.keys()) & set(visible_dict.keys())

    # Crear lista de pares
    pairs = [[lwir_dict[name], visible_dict[name]] for name in common_names]
    print(f"[INFO] Encontrados {len(pairs)} pares de imágenes.")

    save_pkl(pairs, f'{output_data_path}/image_pairs.pkl')
    return pairs