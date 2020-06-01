from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

import scipy.io as sio
import os.path as op
import matplotlib.pyplot as plt

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000
batch_size = 32
gamma = 0.99

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()
if USE_CUDA:
    model = model.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

losses = []
all_rewards = []
episode_reward = 0

loss_list = []
reward_list = []

state = env.reset()

frame_list = random.sample(range(2000, num_frames), 8000)
frame_list.sort()

hiddenLayers = []
state_list = []
action_list = []
reward_frame_list = []
accumulated_reward = []  #####
frame_order = []

epsilon = -1

for frame_idx in range(1, num_frames + 1):

    action = model.act(state, epsilon)

    next_state, reward, done, _ = env.step(action)

    if (frame_idx in frame_list) or (frame_idx < 2000):
        hiddenTensor = model.get_hidden_layer(state)
        temp = hiddenTensor.data.cpu().numpy()
        hiddenLayers.append(temp[0])
        # hiddenLayers.append(hiddenTensor.data.cpu().numpy())
        # hiddenLayers = np.concatenate((hiddenLayers, hiddenTensor.data.cpu().numpy()), axis=0)
        state_list.append(state.squeeze(0))
        action_list.append(action)
        reward_frame_list.append(reward)
        accumulated_reward.append(episode_reward)  #####
        frame_order.append(frame_idx)

    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if frame_idx % 10000 == 0:
        print('#Frame: %d' % frame_idx)

sio.savemat('Results_after_training.mat',
            {'all_rewards': all_rewards, 'hiddenLayers': hiddenLayers, 'state_list': state_list,
             'action_list': action_list, 'reward_frame_list': reward_frame_list,
             'accumulated_reward': accumulated_reward, 'frame_order': frame_order})



