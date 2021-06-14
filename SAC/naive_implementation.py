import gym
import pybullet_envs
import pybullet_envs.bullet as bul
import numpy as np
import torch
from sac_agent import soft_actor_critic_agent, device
from replay_memory import ReplayMemory

seed=0  
env = gym.make('Walker2DBulletEnv-v0')
env.render()
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
max_steps = env._max_episode_steps
print('max_steps: ', max_steps)

batch_size=256 
LEARNING_RATE=0.0003 
start_steps=10000 ## Steps sampling random actions
replay_size=1000000 ## size of replay buffer

agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, \
        hidden_size=256, seed=seed, lr=LEARNING_RATE, gamma=0.99, tau=0.005, alpha=0.2)

memory = ReplayMemory(replay_size)

print('device: ', device)
print('leraning rate: ', LEARNING_RATE)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print('state_dim: ',state_dim, ', action_dim: ', action_dim)

threshold = env.spec.reward_threshold
print('threshold: ', threshold)

def save(agent, directory, filename, episode, reward):
    torch.save(agent.policy.state_dict(), '%s/%s_actor_%s_%s.pth' % (directory, filename, episode, reward))
    torch.save(agent.critic.state_dict(), '%s/%s_critic_%s_%s.pth' % (directory, filename, episode, reward))

import time
from  collections  import deque

def sac_train(max_steps):

    total_numsteps = 0
    updates = 0
    num_episodes = 40000
    updates=0

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = [] 
    num_steps_deque = deque(maxlen=100)
    avg_numm_steps_array = []
    
    for i_episode in range(num_episodes): 
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        for step in range(max_steps):    
            if start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:
                
                # Update parameters of all the networks
                agent.update_parameters(memory, batch_size, updates)

                updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            
            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        min_score = np.min(scores_deque)
        max_score = np.max(scores_deque)
        num_steps_deque.append(episode_steps)
        avg_num_steps = np.mean(num_steps_deque)
        avg_numm_steps_array.append(avg_num_steps)
        
        
        if i_episode % 500 == 0 and i_episode > 0:
            reward_round = round(episode_reward, 2)
            save(agent, 'dir_lr0.0001', 'weights', str(i_episode), str(reward_round))

        s =  (int)(time.time() - time_start)
          
        if i_episode % 10 == 0 and i_episode > 0:  

            print("Ep.: {}, Tot.St.: {}, Avg.Num.St.: {:.1f}, Min-Max.Sc.: ({:.2f}, {:.2f}), Avg.Score: {:.3f}, Time: {:02}:{:02}:{:02}".\
                format(i_episode, total_numsteps, avg_num_steps, min_score, max_score, avg_score, \
                  s//3600, s%3600//60, s%60))

                    
        if (avg_score >= threshold):            
            print('Solved environment with Avg Score:  ', avg_score)
            
            break;
            
    return scores_array, avg_scores_array, avg_numm_steps_array 

scores, avg_scores, avg_numm_steps = sac_train(max_steps=max_steps)

reward_round = round(np.max(scores), 2)
save(agent, 'dir_lr0.0001', 'weights', 'final', str(reward_round))

import matplotlib.pyplot as plt

print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg.Score on 100 episodes")
## plt.plot(np.arange(1, len(avg_numm_steps)+1), avg_numm_steps, label="Avg.Num.Steps on 100 episodes")
plt.legend(bbox_to_anchor=(1.05, 1)) 
plt.ylabel('Score')
plt.xlabel('Episodes #')
plt.show()


fig2 = plt.figure()
ax = fig2.add_subplot(111)
# plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
# plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg.Score on 100 episodes")
plt.plot(np.arange(1, len(avg_numm_steps)+1), avg_numm_steps, label="Avg.Num.Steps on 100 episodes")
plt.legend(bbox_to_anchor=(1.05, 1)) 
plt.ylabel('Num.Steps')
plt.xlabel('Episodes #')
plt.show()

def play(env, agent, num_episodes):
    
    state = env.reset()
    scores_deque = deque(maxlen=100)
    scores = []
    
    for i_episode in range(num_episodes + 1):
        
        state = env.reset()
        score = 0                    
        time_start = time.time()
        
        while True:
            
            action = agent.select_action(state, eval=False)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward 
            state = next_state
    
            if done:
                break
                
        s = (int)(time.time() - time_start)
        
        scores_deque.append(score)
        scores.append(score)    
        
        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'\
                  .format(i_episode, np.mean(scores_deque), score, s//3600, s%3600//60, s%60))

env.close()
