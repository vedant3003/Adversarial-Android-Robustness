import torch
import torch.nn as nn

class MalwareDetectionDNN4L(nn.Module):
    def __init__(self, input_dim=215):
        super(MalwareDetectionDNN4L, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),      
            nn.ReLU(),
            nn.Dropout(0.3),          
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    TOTAL_FEATURES = 215
    
    model = MalwareDetectionDNN4L(input_dim=TOTAL_FEATURES)
    print("Model Architecture Initialized:\n", model)
    
    dummy_input = torch.rand((1, TOTAL_FEATURES)) 
    
    dummy_output = model(dummy_input)
    
    print(f"\nDummy Input Shape: {dummy_input.shape}")
    print(f"Prediction Output (Probability of Malware): {dummy_output.item():.4f}")