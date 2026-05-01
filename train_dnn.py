import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import MalwareDetectionDNN4L

class RealMalwareDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset from {csv_file}...")
        df = pd.read_csv(csv_file, low_memory=False)
        
        df['class'] = df['class'].map({'B': 0, 'S': 1})

        df = df.dropna(subset=['class'])
        
        features_df = df.drop('class', axis=1)
        
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        self.x_data = torch.FloatTensor(features_df.values)
        self.y_data = torch.FloatTensor(df['class'].values).unsqueeze(1)
        self.n_samples = self.x_data.shape[0]
        print(f"Loaded {self.n_samples} total applications.")
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def train_model():
    print("--- Starting DNN Training Phase ---")
    
    dataset = RealMalwareDataset('drebin-215-dataset-5560malware-9476-benign.csv')
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    TOTAL_FEATURES = 215
    model = MalwareDetectionDNN4L(input_dim=TOTAL_FEATURES)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    model.train() 
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'dnn4l_trained_weights.pth')
    print("\n[SUCCESS] Baseline model trained! Saved to 'dnn4l_trained_weights.pth'")

if __name__ == "__main__":
    train_model()