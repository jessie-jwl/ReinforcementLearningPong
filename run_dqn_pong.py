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
from dqn_edit import QLearner, compute_td_loss, ReplayBuffer
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA
from sklearn import preprocessing

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 2000000
batch_size = 32
gamma = 0.95

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)


def main():
    USE_CUDA = torch.cuda.is_available()
    print("Use GPU? ", USE_CUDA)
    model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    if USE_CUDA:
        model = model.cuda()

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 50000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    losses = []
    all_rewards = []
    last_tenK_frames = []
    episode_reward = 0

    state = env.reset()

    for frame_idx in range(1, num_frames + 1):

        epsilon = epsilon_by_frame(frame_idx)

        action = model.act(state, epsilon)
        if frame_idx > (num_frames - 10000):
            last_tenK_frames.append([state, action])

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > replay_initial:
            loss = compute_td_loss(model, batch_size, gamma, replay_buffer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy())

        if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
            print('#Frame: %d, preparing replay buffer' % frame_idx)

        if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
            print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
            print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))

    torch.save(model.state_dict(), "Output/model_states")

    with open("Output/last_tenK_frames", "wb") as outputFile:
        pickle.dump(last_tenK_frames, outputFile)
        outputFile.close()


def test():
    device = torch.device("cuda")
    model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
    model.load_state_dict(torch.load("Output/model_states_2M_PCA"))
    model.to(device)

    features = []

    state = env.reset()
    for _ in range(2000):
        action, feature = model.act(state, 0)
        features.append([state, feature.cpu().data.numpy(), action])
        state, reward, done, _ = env.step(action)

        if done:
            state = env.reset()
    pca(features)
    return


def pca(features):
    tableau10 = {
        'blue': '#507AA6',
        'orange': '#F08E39',
        'red': '#DF585C',
        'teal': '#78B7B2',
        'green': '#5BA053',
        'yellow': '#ECC854',
        'purple': '#AF7BA1',
        'pink': '#FD9EA9',
        'brown': '#9A7460',
        'gray': '#BAB0AC',
        0: '#507AA6',
        1: '#F08E39',
        2: '#DF585C',
        3: '#78B7B2',
        4: '#5BA053',
        5: '#ECC854',
        6: '#AF7BA1',
        7: '#FD9EA9',
        8: '#9A7460',
        9: '#BAB0AC',
    }

    print(len(features))

    for i in range(20, 100):
        frame, _, _ = features[i]
        plt.imshow(frame[0])
        plt.savefig("Output/play/Frame_" + str(i))

    # Below PCA stuff.
    last_df = pd.DataFrame(features, columns=['Frame', 'Feature', 'Action'])
    last_df['Frame'] = last_df['Frame'].apply(lambda x: x[0].reshape(-1))

    last_frame = np.row_stack(last_df['Feature'].to_numpy())
    last_frame = preprocessing.scale(last_frame)

    last_pca = PCA(n_components=2).fit_transform(last_frame)

    print(len(set(last_df['Action'])))
    plt.figure()
    lw = 2
    colors = [tableau10['orange'], tableau10['green'], tableau10['blue'], tableau10['purple'], tableau10['teal'], 'tomato']
    for i in (0, 1, 2, 3, 4, 5):
        plt.scatter(last_pca[last_df['Action'] == i, 0], last_pca[last_df['Action'] == i, 1], color=colors[i], alpha=.8, lw=lw, label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    plt.show()


if __name__ == '__main__':
    test()
