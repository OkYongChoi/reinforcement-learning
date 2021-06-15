#%%import numpy as np
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import csv

import gym
import pybullet_envs.bullet as bul

import math
import time
from td3_agent import ReplayBuffer, TD3Agent
from collections import deque


start_timestep=1e4

std_noise=0.04

env = gym.make('Walker2DBulletEnv-v0')
env.render()

# Set seeds
seed = 12345
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
threshold = env.spec.reward_threshold

print('start_dim: ', state_dim, ', action_dim: ', action_dim)
print('max_action: ', max_action, ', threshold: ', threshold, ', std_noise: ', std_noise)

agent = TD3Agent(state_dim, action_dim, max_action)

# Twin Delayed Deep Deterministic (TD3) policy gradient algorithm
def td3_train(n_episodes=10000, save_every=10, print_env=10):

    scores_deque = deque(maxlen=10)
    scores_array = []
    avg_scores_array = []    
    best_score = -math.inf

    time_start = time.time()                    # Init start time
    replay_buf = ReplayBuffer()                 # Init ReplayBuffer
    
    timestep_after_last_save = 0
    total_timesteps = 0
    
    low = env.action_space.low
    high = env.action_space.high
    
    print(f'Low in action space: {low}, High: {high}, Action_dim: {action_dim}')
            
    for i_episode in range(1, n_episodes+1):
        
        timestep = 0
        total_reward = 0
        
        # Reset environment
        state = env.reset()
        done = False

        while done is False:
            
            # Select action randomly or according to policy
            if total_timesteps < start_timestep:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
                if std_noise != 0: 
                    shift_action = np.random.normal(0, std_noise, size=action_dim)
                    action = (action + shift_action).clip(low, high)
            
            # Perform action
            new_state, reward, done, _ = env.step(action) 
            done_bool = 0 if timestep + 1 == env._max_episode_steps else float(done)
            total_reward += reward                          # full episode reward

            # Store every timestep in replay buffer
            replay_buf.add((state, new_state, action, reward, done_bool))
            state = new_state

            timestep += 1     
            total_timesteps += 1
            timestep_after_last_save += 1

        scores_deque.append(total_reward)
        scores_array.append(total_reward)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        if total_reward > best_score:
            best_score = total_reward
            agent.save('dir_Walker2D_002', 'td3_best')

        # train_by_episode(time_start, i_episode) 
        s = (int)(time.time() - time_start)
        if i_episode % print_env == 0 or (len(scores_deque) == 100 and avg_score > threshold):
            print('Ep. {}, Timestep {}, Ep.Timesteps {}, Score: {:.2f}, Avg.Score: {:.2f}, Best.Score: {:.2f}, Time: {:02}:{:02}:{:02} '\
                .format(i_episode, total_timesteps, timestep, \
                        total_reward, avg_score, best_score, s//3600, s%3600//60, s%60))     

        agent.train(replay_buf, timestep)
        
        if avg_score >= threshold:
            print('Environment solved with Average Score: ',  avg_score )

        # Write to tensorboard
        writer.add_scalar("average return", total_reward, i_episode)

    return scores_array, avg_scores_array

scores, avg_scores = td3_train()

agent.save('dir_Walker2D_002', 'td3_last')
writer.flush()
writer.close()

with open('td3_scores_1000episodes.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(scores)

#%%