# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:47:29 2025

@author: Altinses, M.Sc.

To-Do:
    - Nothing...
"""

# %% imports

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import os
from PIL import Image

from dataset import generate_synthetic_cost_data
from config_plots import configure_plt
from network import Autoencoder

# %% config

np.random.seed(42)
torch.manual_seed(42)

configure_plt()

# %% generate data

df, labels = generate_synthetic_cost_data()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# DataLoader 
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %% Isolation Forest Model

print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)

iso_scores = -iso_forest.decision_function(X_test)  # Umso höher, desto anomaler
iso_pred = (iso_scores > np.percentile(iso_scores, 95)).astype(int)  # Top 5% als Anomalien

print("\nIsolation Forest Ergebnisse:")
print(classification_report(y_test, iso_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, iso_scores):.4f}")

# %% autoencoder

input_dim = X_train.shape[1]
autoencoder = Autoencoder(input_dim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

# Training
print("\nTraining Autoencoder...")
n_epochs = 50
train_losses = []

os.makedirs("training_progress", exist_ok=True)

for epoch in range(n_epochs):
    epoch_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        
        # Forward pass
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    if epoch % 1 == 0 or epoch == n_epochs - 1:
        with torch.no_grad():
            test_outputs = autoencoder(X_test_tensor)
            test_loss = criterion(test_outputs, X_test_tensor).item()
            
            # Rekonstruktionsfehler berechnen
            errors = torch.mean((X_test_tensor - test_outputs)**2, dim=1).numpy()
            
            plt.figure(figsize=(6, 3))
            
            # Rekonstruktionsfehler für normale Daten und Anomalien
            plt.subplot(1, 2, 1)
            sns.kdeplot(errors[y_test == 0], label="Normal", shade=True)
            sns.kdeplot(errors[y_test == 1], label="Anomaly", shade=True)
            plt.xlabel("Error")
            plt.ylabel("Density")
            plt.ylim(0,1000)
            plt.grid()
            plt.legend()
            
            # Loss-Verlauf
            plt.subplot(1, 2, 2)
            plt.plot(train_losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.ylim(0,0.018)
            plt.grid()
            
            plt.tight_layout()
            plt.savefig(f"training_progress/epoch_{epoch+1}.png", dpi = 150)
            plt.close()

# %% Evaluation

with torch.no_grad():
    test_outputs = autoencoder(X_test_tensor)
    errors = torch.mean((X_test_tensor - test_outputs)**2, dim=1).numpy()
    
# Threshold (95% Quantil)
threshold = np.percentile(errors, 95)
ae_pred = (errors > threshold).astype(int)

print(classification_report(y_test, ae_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, errors):.4f}")

print("Isolation Forest ROC-AUC:", roc_auc_score(y_test, iso_scores))
print("Autoencoder ROC-AUC:", roc_auc_score(y_test, errors))

# %% GIF
def create_gif(image_folder, output_gif, duration=200):
    images = []
    file_names = sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for file_name in file_names:
        if file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))
    
    images[0].save(output_gif, save_all=True, append_images=images[1:], 
                   optimize=False, duration=duration, loop=0)

create_gif("training_progress", "training_progress.gif")

print("\nTraining GIF wurde als 'training_progress.gif' gespeichert.")

# %% plots

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
sns.scatterplot(x=range(len(iso_scores)), y=iso_scores, hue=y_test, palette={0: 'blue', 1: 'red'}, legend='full')
plt.axhline(y=np.percentile(iso_scores, 95), color='green', linestyle='--', label='Threshold')
plt.title("Anomalie-Scores")
plt.xlabel("Samples")
plt.ylabel("Anomalie-Score")
plt.legend(labels=["Normal", "Anomaly"])

ax1 = plt.gca()
handles, labels = ax1.get_legend_handles_labels()
new_labels = ['Normal', 'Anomaly', 'Threshold']
new_handles = [handles[0], handles[1], handles[2]]
ax1.legend(handles=new_handles, labels=new_labels, loc='upper left')
plt.grid()
plt.ylim(-0.2,0.5)

plt.subplot(1, 2, 2)
sns.scatterplot(x=range(len(errors)), y=errors, hue=y_test, palette={0: 'blue', 1: 'red'}, legend='full')
plt.title("Reconstruction")
plt.xlabel("Samples")
plt.ylabel("Error")
ax2 = plt.gca()
handles, labels = ax2.get_legend_handles_labels()
new_labels = ['Normal', 'Anomaly']
new_handles = [handles[0], handles[1]]

# Create a new legend with the desired labels and handles
ax2.legend(handles=new_handles, labels=new_labels, loc='upper left')
plt.grid()

plt.tight_layout()
plt.savefig("anomaly_detection_comparison.png")
plt.show()