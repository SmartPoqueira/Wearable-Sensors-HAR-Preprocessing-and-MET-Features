import numpy as np
import pandas as pd
import os
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier

RANDOM_SEED = 5
path = './data/'
from data_helper import ensure_data_and_features
foldersMIOS = ensure_data_and_features(path)

# Subset of Activities for Table V (likely)
# Drawing, Dishes, Regular Walk, Up/Down Stairs
activitySet = ['d', 'wd', 'wr', 'ds', 'us']

featuresSel = ['x', 'y', 'z', 'smv', 'smvfil', 'ratio', 'HRR', 'hr', 'xfil', 'yfil', 'zfil']
featuresScalatedCut = 'featur-scale-cut-mios.csv'

def run_tabnet_subset():
    logocv = LeaveOneGroupOut()
    dataset = []
    groups = []
    i = 0

    # Load data
    print("Loading data for subset...")
    for fichero in os.listdir(path):
        if fichero in foldersMIOS:
            file_path = os.path.join(path, fichero, featuresScalatedCut)
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path)
                # Filter for subset
                temp_df = temp_df.loc[temp_df['activity'].isin(activitySet)]
                
                if not temp_df.empty:
                    dataset.append(temp_df)
                    
                    personlen = len(temp_df)
                    value = i
                    group = np.empty(personlen)
                    group.fill(value)
                    groups = np.concatenate((groups, group), axis=None)
                    i += 1
            else:
                print(f"Warning: File not found {file_path}")

    if not dataset:
        print("No data loaded.")
        return

    frame = pd.concat(dataset)
    X = frame.loc[:, featuresSel].values
    y = frame.loc[:, "activity"].values
    
    # Encode targets
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Classes: {le.classes_}")

    accuracies = []
    
    # Cross-validation
    print("Starting Cross-Validation (Subset)...")
    for train_index, test_index in logocv.split(X, y_encoded, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # TabNet Classifier
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=0,
            seed=RANDOM_SEED
        )
        
        # Fit
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=100,
            patience=20,
            batch_size=1024, 
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Predict
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"Fold Accuracy: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "="*30)
    print(f"TabNet Results (Subset: {activitySet})")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Std Deviation: {std_acc:.4f}")
    print("="*30)
    
    return accuracies

if __name__ == "__main__":
    run_tabnet_subset()
