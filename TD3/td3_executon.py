
import numpy as np
import torch

import gym
import pybullet_envs.bullet.kuka_diverse_object_gym_env

import time
from td3_agent import TD3Agent
from collections import deque



start_timestep=1e4

std_noise=0.02

env = gym.make('Walker2DBulletEnv-v0')
env.render()

# Set seeds
seed = 12345
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
threshold = env.spec.reward_threshold

print('start_dim: ', state_dim, ', action_dim: ', action_dim)
print('max_action: ', max_action, ', threshold: ', threshold, ', std_noise: ', std_noise)

agent = TD3Agent(state_dim, action_dim, max_action)
agent.load('dir_Walker2D_002', 'td3_best', )

def play(env, agent, n_episodes):
    scores_deque = deque(maxlen=10)
    scores = []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()        
        score = 0
        
        time_start = time.time()
        
        while True:
            action = agent.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break 

        s = (int)(time.time() - time_start)
        
        scores_deque.append(score)
        scores.append(score)
        
        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'\
                  .format(i_episode, np.mean(scores_deque), score, s//3600, s%3600//60, s%60))  

play(env=env, agent=agent, n_episodes=5)