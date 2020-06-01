import itertools
import numpy as np
import scipy.io as sio
import os.path as op
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA

import matplotlib.patheffects as PathEffects
import seaborn as sns

# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

#Load Data

R = np.array([])
L = np.array([])
if op.isfile('Results.mat'):
    loadData = sio.loadmat('Results.mat')

    R = loadData['reward_list']
    R.shape = (-1, 1)
    L = loadData['loss_list']
    L.shape = (-1, 1)

all_r = np.array([])
hLayers = np.array([])
a_list = np.array([])
s_list = np.array([])
reward_frame = np.array([])
acc_reward = np.array([])
f_order = np.array([])
if op.isfile('Results_after_training.mat'):
    loadData = sio.loadmat('Results_after_training.mat')

    all_r = loadData['all_rewards']
    hLayers = loadData['hiddenLayers']
    a_list = loadData['action_list']
    s_list = loadData['state_list']
    reward_frame = loadData['reward_frame_list']
    acc_reward = loadData['accumulated_reward']
    f_order = loadData['frame_order']

a_list = a_list[0]
reward_frame = reward_frame[0]
acc_reward = acc_reward[0]
f_order = f_order[0]
# hLayers = np.array(hLayers)
# a_list = np.array(a_list)


#Plot Reward and Loss

t = np.arange(1, R.size + 1, 1)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('frames x 10^4')
ax1.set_ylabel('Reward', color=color)
ax1.plot(t, R, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
ax2.plot(t, L, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig('reward_loss_plot.png', bbox_inches='tight', dpi=300)

#Data Vusualization
t_hLayers = hLayers[2000:3000, :]
t_a_list = a_list[2000:3000]
t_r_list = reward_frame[2000:3000]

#Try three different method listed in the assignment
Layer_embedded = PCA(n_components=2).fit_transform(t_hLayers)
Layer_embedded = LocallyLinearEmbedding(n_components=2).fit_transform(t_hLayers)
Layer_embedded = Isomap(n_components=2).fit_transform(t_hLayers)
# Layer_embedded = TSNE(n_components=2).fit_transform(t_hLayers)

num_classes = len(np.unique(t_a_list))

my_cmap = plt.cm.get_cmap('RdBu_r')

#Create 2D scatter plot
f = plt.figure(figsize=(10, 10))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(Layer_embedded[:, 0], Layer_embedded[:, 1], lw=0, s=40, c=t_a_list, cmap=my_cmap)
plt.colorbar(sc)
plt.show()
