
import numpy as np

def scaleAffineTransform(T, average_fraction = 0.0919):
    T = np.array(T)
    A = T[:2, :2]
    t = T[:2, 2]
    
    I = np.eye(2)
    A_scaled = I + (A - I) * average_fraction
    t_scaled = t * average_fraction
    
    T_scaled = np.eye(3)[:2, :]  # Create a 2x3 matrix
    T_scaled[:2, :2] = A_scaled
    T_scaled[:2, 2] = t_scaled
    
    return T_scaled

def applyTransformation(point, transformation_matrix):
    point = np.array(point, dtype=np.float32)
    if transformation_matrix.shape != (2, 3):
        raise ValueError(f"Transform matriz should have a shape of (2,3); instead it has {transformation_matrix.shape}")
    transformed_point = np.dot(transformation_matrix[:, :2], point) + transformation_matrix[:, 2]
    return transformed_point

def invertAffineTransform(transform):
    transform_homogeneous = np.vstack([transform, [0, 0, 1]])
    inverted_homogeneous = np.linalg.inv(transform_homogeneous)
    inverted_transform = inverted_homogeneous[:2, :]
    return inverted_transform