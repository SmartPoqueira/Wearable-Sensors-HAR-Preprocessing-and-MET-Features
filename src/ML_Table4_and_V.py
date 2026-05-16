import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
from tab_transformer_pytorch import TabTransformer
import warnings

warnings.filterwarnings("ignore")

RANDOM_SEED = 5
path = './codigo/clean/Data/'
featuresScalatedCut = 'featur-scale-cut-mios.csv'

foldersMIOS = ['F-014', 'F-046', 'F-047', 'M-003', 'M-004', 'M-007', 'M-008']

feature_configs = {
    'x, y, z': ['x', 'y', 'z'],
    'x, y, z, SVMg': ['x', 'y', 'z', 'smv'],
    'xfil, yfil, zfil, SVMg_fil': ['xfil', 'yfil', 'zfil', 'smvfil'],
    'SVMg_fil, RUF': ['smvfil', 'ratio'],
    'SVMg_fil, RUF, %HRR': ['smvfil', 'ratio', 'HRR'],
    'all': ['x', 'y', 'z', 'smv', 'xfil', 'yfil', 'zfil', 'smvfil', 'ratio', 'hr', 'HRR']
}

models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=RANDOM_SEED)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('DT-CART', DecisionTreeClassifier(random_state=RANDOM_SEED)), 
    ('NB', GaussianNB()),
    ('TabNet', None),
    ('TabTransformer', None)
]

class TabTransformerClassifier:
    def __init__(self, categories, num_continuous, dim=32, depth=6, heads=8, dim_head=16, mlp_hidden_mults=(4, 2), mlp_act=nn.ReLU(), dropout=0.1, lr=1e-3, epochs=50, batch_size=256):
        self.categories = categories 
        self.num_continuous = num_continuous
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_hidden_mults = mlp_hidden_mults
        self.mlp_act = mlp_act
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TabTransformer(
            categories=categories, 
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dim_out=len(np.unique(categories)) if categories else 1, 
            mlp_hidden_mults=mlp_hidden_mults,
            mlp_act=mlp_act,
        ).to(self.device)

    def fit(self, X, y):
        num_classes = len(np.unique(y))
        self.model = TabTransformer(
            categories=(), 
            num_continuous=self.num_continuous,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            dim_out=num_classes,
            mlp_hidden_mults=self.mlp_hidden_mults,
            mlp_act=self.mlp_act,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(torch.empty((batch_X.shape[0], 0), dtype=torch.long, device=self.device), batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_X, in dataloader:
                pred = self.model(torch.empty((batch_X.shape[0], 0), dtype=torch.long, device=self.device), batch_X)
                preds.append(torch.argmax(pred, dim=1).cpu().numpy())
        return np.concatenate(preds)

def get_tabnet_model():
    return TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=0,
        seed=RANDOM_SEED
    )

def run_experiment(activity_set, table_name):
    print(f"\nGenerating {table_name}...")
    print("-" * 100)
    
    dataset = []
    groups = []
    i = 0
    
    for fichero in os.listdir(path):
        if fichero in foldersMIOS:
            file_path = os.path.join(path, fichero, featuresScalatedCut)
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path)
                temp_df = temp_df.loc[temp_df['activity'].isin(activity_set)]
                
                if not temp_df.empty:
                    dataset.append(temp_df)
                    personlen = len(temp_df)
                    value = i
                    group = np.empty(personlen)
                    group.fill(value)
                    groups = np.concatenate((groups, group), axis=None)
                    i += 1
    
    if not dataset:
        print("No data found!")
        return

    frame = pd.concat(dataset)
    y = frame.loc[:, "activity"].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logocv = LeaveOneGroupOut()
    
    results_table = {}
    
    for config_name, features in feature_configs.items():
        print(f"  Processing features: {config_name}")
        X = frame.loc[:, features].values
        
        for model_name, model_inst in models:
            if model_name not in results_table:
                results_table[model_name] = {}
            
            if model_name == 'TabNet':
                accuracies = []
                for train_index, test_index in logocv.split(X, y_encoded, groups):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
                    
                    clf = get_tabnet_model()
                    clf.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        eval_name=['train', 'valid'],
                        eval_metric=['accuracy'],
                        max_epochs=50, 
                        patience=10,
                        batch_size=1024, 
                        virtual_batch_size=128,
                        num_workers=0,
                        drop_last=False
                    )
                    preds = clf.predict(X_test)
                    accuracies.append(accuracy_score(y_test, preds))
                
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
            
            elif model_name == 'TabTransformer':
                accuracies = []
                for train_index, test_index in logocv.split(X, y_encoded, groups):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
                    
                    clf = TabTransformerClassifier(
                        categories=(), 
                        num_continuous=X.shape[1],
                        dim=32, 
                        depth=6, 
                        heads=8, 
                        dim_head=16,
                        mlp_hidden_mults=(4, 2),
                        dropout=0.1,
                        lr=1e-3,
                        epochs=50,
                        batch_size=256
                    )
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    accuracies.append(accuracy_score(y_test, preds))
                
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

            else:
                cv_results = cross_val_score(model_inst, X, y, cv=logocv, groups=groups, scoring='accuracy')
                mean_acc = cv_results.mean()
                std_acc = cv_results.std()
            
            results_table[model_name][config_name] = f"{mean_acc:.2f} ± {std_acc:.2f}"

    print("\n" + "=" * 120)
    print(f"{table_name}")
    print("=" * 120)
    
    headers = ["Model"] + list(feature_configs.keys())
    header_row = "{:<15} | " + " | ".join([f"{{:<{len(h)+2}}}" for h in feature_configs.keys()])
    print(header_row.format(*headers))
    print("-" * 120)
    
    for model_name, _ in models:
        row = [model_name]
        for config_name in feature_configs.keys():
            row.append(results_table[model_name][config_name])
        
        row_fmt = "{:<15} | " + " | ".join([f"{{:<{len(h)+2}}}" for h in feature_configs.keys()])
        print(row_fmt.format(*row))
    print("=" * 120 + "\n")

from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    all_activities = ['lie', 'r', 'd', 'wd', 'wp', 'ws', 'wr', 'wf', 'ds', 'us']
    run_experiment(all_activities, "Table IV: All Activities")
    
    subset_activities = ['d', 'wd', 'wr', 'ds', 'us']
    run_experiment(subset_activities, "Table V: Subset (Draw, Dishes, Walk, Stairs)")
