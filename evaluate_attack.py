import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from model import MalwareDetectionDNN4L
from env import MalwareEvasionEnv 

def load_malware_only(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    malware_df = df[df['class'] == 'S'].copy()
    features_df = malware_df.drop('class', axis=1)
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return features_df.values.astype(np.int8)

def evaluate_agent():
    TOTAL_FEATURES = 215
    TEST_EPISODES = 1000 
    
    print("--- Starting Final Evaluation ---")
    
    target_model = MalwareDetectionDNN4L(input_dim=TOTAL_FEATURES)
    target_model.load_state_dict(torch.load('dnn4l_trained_weights.pth'))
    target_model.eval()
    
    malware_data = load_malware_only('drebin-215-dataset-5560malware-9476-benign.csv')
    env = MalwareEvasionEnv(target_model=target_model, malware_dataset=malware_data, max_perturbations=10, num_features=TOTAL_FEATURES)
    
    print("Loading trained PPO Agent...")
    rl_agent = PPO.load("ppo_malware_evasion_agent")
    
    success_count = 0
    total_perturbations_on_success = 0
    
    print(f"\nAttacking {TEST_EPISODES} malware samples...")
    for i in range(TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _states = rl_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            
        if info.get('fooled', False):
            success_count += 1
            total_perturbations_on_success += info.get('perturbation_count', 0)
            
    asr = (success_count / TEST_EPISODES) * 100
    avg_perturbations = (total_perturbations_on_success / success_count) if success_count > 0 else 0
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Total Malware Tested: {TEST_EPISODES}")
    print(f"Successful Evasions:  {success_count}")
    print(f"Attack Success Rate:  {asr:.2f}%")
    if success_count > 0:
        print(f"Avg Permissions Added to Fool Model: {avg_perturbations:.2f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_agent()