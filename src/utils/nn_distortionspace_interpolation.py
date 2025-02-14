#!/usr/bin/env python3
# encoding: utf-8

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from utils.get_data import getPointData
from utils.plot_utils import estimate_distortion_center


class VectorFieldModel(nn.Module):
    def __init__(self):
        super(VectorFieldModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.model(x)

class RBFNet(nn.Module):
    def __init__(self, indim, num_centers, outdim):
        super(RBFNet, self).__init__()
        self.indim = indim
        self.num_centers = num_centers
        self.outdim = outdim
        self.centers = nn.Parameter(torch.Tensor(num_centers, indim))
        self.sigmas = nn.Parameter(torch.Tensor(num_centers))
        self.linear = nn.Linear(num_centers, outdim)
        self.init_centers_sigmas()
    
    def init_centers_sigmas(self):
        nn.init.uniform_(self.centers, -1, 1)
        nn.init.constant_(self.sigmas, 0.1)

    def rbf(self, x, center, sigma):
        return torch.exp(-sigma * torch.sum((x - center)**2, axis=1))
    
    def forward(self, x):
        # Compute RBF activations
        phi = torch.stack([self.rbf(x, c, s) for c, s in zip(self.centers, self.sigmas)])
        phi = phi.t()
        return self.linear(phi)
    
# Función de entrenamiento
def train_model(data_train, data_val = None, epochs=500, patience=30, fold_log = ""):
    X_train, y_train = data_train
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    if data_val is not None:
        X_val, y_val = data_val
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32)
    else:
        val_loader = train_loader
        
    model = VectorFieldModel()
    # model = RBFNet(indim=3, num_centers=1, outdim=2)
    # Weight_decay -> penalieze big weights making model simpler and with better generalization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    with tqdm(range(epochs), desc=f"{fold_log}Train", unit="epoch") as pbar:
        for epoch in pbar:
            model.train()
            # Entrenamiento
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluación (sin gradientes)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()

            # Promedio de la pérdida de validación
            val_loss /= len(val_loader)
            pbar.set_postfix(val_loss=val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        
            if epochs_without_improvement >= patience:
                pbar.set_postfix_str(f"Stopping early at epoch {epoch + 1}/{epochs}. Best result at epoch: {epoch+1-patience}. Best val_loss: {best_val_loss:4f}")
                # print(f"Early stopping at epoch {epoch + 1}")
                pbar.close()
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    val_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    val_loss /= len(val_loader)
    
    cv_score = (val_loss)
    print(f"Validation loss: {val_loss:.4f}")
    return model, cv_score


def interpolateDistortionGridNN(image_points_lwir, image_points_rgb, image_size, grid_size = 10):
    # Función para calcular la distancia al centro de distorsión estimado
    def distance_to_center(point, distortion_center):
        return np.sqrt((point[0] - distortion_center[0])**2 + (point[1] - distortion_center[1])**2)

    print(f"{image_points_lwir.shape = }; {image_points_rgb.shape = }")
    X , Y = image_points_lwir, image_points_rgb, 

    # Add distance to distortion center as a feature for training
    distortion_center = estimate_distortion_center(X, Y)
    distances = np.array([distance_to_center(point, distortion_center) for point in X])
    X_with_distance = np.column_stack((X, distances))


    # Normalización de datos
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_with_distance)
    y_scaled = scaler_y.fit_transform(Y)

    # Tensors to pytorch format
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)

    # Validación cruzada
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = []
    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]
        
        model, cv_score = train_model((X_train, y_train), (X_val, y_val), fold_log=f"Fold {fold+1}/{n_splits}. ")
        cv_scores.append(cv_score)
        
    print(f"Mean cross-validation score: {np.mean(cv_scores):.4f}")

    # Final training with all data
    final_model, cv_score = train_model((X_tensor, y_tensor), fold_log="Final training.")

    # Predict new points based on model
    def predict_vector(point, distortion_center):
        distance = distance_to_center(point, distortion_center)
        input_data = np.array([[*point, distance]])
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)
        
        final_model.eval()
        with torch.no_grad():
            prediction_scaled = final_model(input_tensor).numpy()
        
        prediction = scaler_y.inverse_transform(prediction_scaled)
        return prediction[0]

    # Ejemplo de predicción
    interpolated_x = []
    interpolated_y = []
    for x_coord in range(0, grid_size):
        for y_coord in range(0, grid_size):
            new_point = (int(image_size[0]/grid_size*x_coord), int(image_size[1]/grid_size*y_coord))
            interpolated_x.append(new_point)
            interpolated_y.append(predict_vector(new_point, distortion_center))
    
    print(f"Interpolated grid with {len(interpolated_y)} points.")
    return np.array(interpolated_x, dtype='float32'), np.array(interpolated_y, dtype='float32')
