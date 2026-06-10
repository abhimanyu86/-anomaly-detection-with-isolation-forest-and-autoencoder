# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:21:00 2025

@author: Altinses
"""

#%% imports

import numpy as np
import pandas as pd

# %%

def generate_synthetic_cost_data(n_samples=10000, anomaly_ratio=0.05):
    """Generiert synthetische Kostenabweichungsdaten mit Anomalien"""
    # Normale Daten (multivariate Normalverteilung)
    n_normal = int(n_samples * (1 - anomaly_ratio))
    normal_data = np.random.normal(loc=100, scale=10, size=(n_normal, 5))
    
    n_anomaly = n_samples - n_normal
    anomaly1 = np.random.normal(loc=200, scale=5, size=(n_anomaly//2, 5))
    anomaly2 = np.random.uniform(low=20, high=40, size=(n_anomaly//2, 5))
    anomaly_data = np.vstack([anomaly1, anomaly2])
    
    data = np.vstack([normal_data, anomaly_data])
    labels = np.array([0]*n_normal + [1]*n_anomaly)
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    features = ['Materialkosten', 'Arbeitskosten', 'Logistikkosten', 'Energiekosten', 'Wartungskosten']
    
    return pd.DataFrame(data, columns=features), labels