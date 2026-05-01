import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random

class MalwareEvasionEnv(gym.Env):
    def __init__(self, target_model, malware_dataset, max_perturbations=10, num_features=215):
        super(MalwareEvasionEnv, self).__init__()
        
        self.target_model = target_model
        self.malware_dataset = malware_dataset 
        self.max_perturbations = max_perturbations
        self.num_features = num_features
        
        self.action_space = spaces.Discrete(self.num_features)
        self.observation_space = spaces.MultiBinary(self.num_features)
        
        self.current_state = None
        self.original_score = None
        self.perturbation_count = 0
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        random_malware = random.choice(self.malware_dataset)
        self.current_state = np.copy(random_malware).astype(np.int8)
        self.perturbation_count = 0
        self.step_count = 0

        with torch.no_grad():
            tensor_state = torch.FloatTensor(self.current_state).unsqueeze(0)
            self.original_score = self.target_model(tensor_state).item()
            
        return self.current_state, {}

    def step(self, action):
        self.step_count += 1  
        
        if self.current_state[action] == 0:
            self.current_state[action] = 1
            self.perturbation_count += 1
            
        with torch.no_grad():
            tensor_state = torch.FloatTensor(self.current_state).unsqueeze(0)
            new_score = self.target_model(tensor_state).item()
            
        reward = self.original_score - new_score 
        self.original_score = new_score
        
        fooled = new_score < 0.5
        terminated = bool(fooled)
        
        truncated = bool(self.step_count >= self.max_perturbations) 
        
        info = {'fooled': fooled, 'perturbation_count': self.perturbation_count}
        
        return self.current_state, float(reward), terminated, truncated, info