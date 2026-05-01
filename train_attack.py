import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from model import MalwareDetectionDNN4L
from env import MalwareEvasionEnv 

def load_malware_only(csv_file):
    print("Loading malware samples for the RL Environment...")
    df = pd.read_csv(csv_file, low_memory=False)
    malware_df = df[df['class'] == 'S'].copy()
    features_df = malware_df.drop('class', axis=1)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return features_df.values.astype(np.int8)

def train_ppo_agent():
    TOTAL_FEATURES = 215
    
    print("--- Initializing RL Attack Simulation ---")
    target_model = MalwareDetectionDNN4L(input_dim=TOTAL_FEATURES)
    try:
        target_model.load_state_dict(torch.load('dnn4l_trained_weights.pth'))
        print("[SUCCESS] Loaded trained DNN weights.")
    except FileNotFoundError:
        print("[ERROR] Could not find 'dnn4l_trained_weights.pth'. Run train_dnn.py first!")
        return

    target_model.eval() 
    
    malware_data = load_malware_only('drebin-215-dataset-5560malware-9476-benign.csv')

    env = MalwareEvasionEnv(target_model=target_model, malware_dataset=malware_data, max_perturbations=10, num_features=TOTAL_FEATURES)
    
    check_env(env, warn=True)

    print("\n[START] Training PPO Agent on real malware...")
    rl_agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    rl_agent.learn(total_timesteps=20000)
    
    rl_agent.save("ppo_malware_evasion_agent")
    print("\n[SUCCESS] PPO Agent Training Complete and Saved!")

if __name__ == "__main__":
    train_ppo_agent()