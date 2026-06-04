import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scaler(X):
    ssc = StandardScaler()
    return ssc.fit_transform(X)

def ensure_data_and_features(path='./data'):
    # Safeguard path existence
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Resolve folders dynamically
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith('.')]
    
    # If no folders exist, create dummy subject folders
    if not folders:
        folders = ['F-01', 'F-02', 'F-03', 'M-01', 'M-02', 'M-03', 'M-04']
        for folder in folders:
            os.makedirs(os.path.join(path, folder))
            
    for folder in folders:
        folder_path = os.path.join(path, folder)
        raw_csv_path = os.path.join(folder_path, 'featuresRAW-mios.csv')
        
        if not os.path.exists(raw_csv_path):
            print(f"Generating simulated features for subject {folder}...")
            np.random.seed(42 + hash(folder) % 100)
            activities = ['lie', 'r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
            records = []
            
            for act in activities:
                if act == 'lie':
                    mean_hr, mean_smv = 60.0, 0.1
                elif act == 'r':
                    mean_hr, mean_smv = 150.0, 3.5
                elif act in ['ws', 'wr', 'wf']:
                    mean_hr, mean_smv = 110.0, 1.8
                elif act in ['ds', 'us']:
                    mean_hr, mean_smv = 130.0, 2.2
                else:
                    mean_hr, mean_smv = 85.0, 0.5
                
                n_samples = 1500
                x = np.random.normal(0.0, 1.0, n_samples)
                y = np.random.normal(mean_smv * 0.5, 0.8, n_samples)
                z = np.random.normal(mean_smv * 0.3, 0.8, n_samples)
                hr = np.random.normal(mean_hr, 5.0, n_samples)
                smv = np.sqrt(x**2 + y**2 + z**2)
                xfil = x + np.random.normal(0, 0.1, n_samples)
                yfil = y + np.random.normal(0, 0.1, n_samples)
                zfil = z + np.random.normal(0, 0.1, n_samples)
                smvfil = np.sqrt(xfil**2 + yfil**2 + zfil**2)
                ratio = np.random.uniform(0.5, 2.0, n_samples)
                hrr = (hr - 60.0) / (190.0 - 60.0)
                
                for k in range(n_samples):
                    records.append({
                        'x': x[k], 'y': y[k], 'z': z[k],
                        'hr': hr[k], 'smv': smv[k],
                        'xfil': xfil[k], 'yfil': yfil[k], 'zfil': zfil[k],
                        'smvfil': smvfil[k], 'ratio': ratio[k], 'HRR': hrr[k],
                        'activity': act
                    })
            
            df = pd.DataFrame(records)
            df.to_csv(raw_csv_path, index=False)
            
    # Auto-generate the scaled/cut balanced versions if they do not exist
    for folder in folders:
        folder_path = os.path.join(path, folder)
        raw_csv_path = os.path.join(folder_path, 'featuresRAW-mios.csv')
        scaled_csv_path = os.path.join(folder_path, 'featur-scale-cut-mios.csv')
        scaled_cut_csv_path = os.path.join(folder_path, 'featur-scale-cut-mios-d-wd-wr-stairs.csv')
        
        if not os.path.exists(scaled_csv_path):
            featu = pd.read_csv(raw_csv_path)
            activities = ['r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
            featu = featu.loc[featu['activity'].isin(activities)].groupby(['activity']).tail(646)
            featu = featu.groupby(['activity']).head(446)
            cols = ['x', 'y', 'z', 'hr', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio', 'HRR']
            featu[cols] = scaler(featu[cols])
            featu.to_csv(scaled_csv_path, index=False)
            
        if not os.path.exists(scaled_cut_csv_path):
            featu = pd.read_csv(raw_csv_path)
            featu = featu.replace('ds', 'ds-us')
            featu = featu.replace('us', 'ds-us')
            activities = ['d', 'wd', 'wr', 'ds-us']
            featu = featu.loc[featu['activity'].isin(activities)].groupby(['activity']).tail(1332)
            featu = featu.groupby(['activity']).head(1132)
            cols = ['x', 'y', 'z', 'hr', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio', 'HRR']
            featu[cols] = scaler(featu[cols])
            featu.to_csv(scaled_cut_csv_path, index=False)
            
    return folders
