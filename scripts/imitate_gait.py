# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, BatchNormalization
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math
import scipy.io

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()

# Load gait parameters
gait_path = '/home/ubuntu/still_params.mat'
gait_params = scipy.io.loadmat(gait_path)
tau_coeffs = gait_params['tau_coeffs_wrt_time'][0]

qd_coeffs = gait_params['qd_coeffs_wrt_tau']
qd_coeffs_wrt_time = gait_params['qd_coeffs_wrt_time']
dot_qd_coeffs = gait_params['dot_qd_coeffs_wrt_tau']

# order of these coeffs (according to Wen-Loong) is:
# 0. nonstance ankle
# 1. nonstance knee
# 2. nonstance hip
# 3. stance hip
# 4. stance knee
# 5. stance ankle

torso_coeffs = gait_params['torso_coeffs_wrt_tau'][0]

sf_talus_X_coeffs = gait_params['sf_talus_X_coeffs_wrt_tau'][0]
sf_talus_Y_coeffs = gait_params['sf_talus_Y_coeffs_wrt_tau'][0]
nsf_talus_X_coeffs = gait_params['nsf_talus_X_coeffs_wrt_tau'][0]
nsf_talus_Y_coeffs = gait_params['nsf_talus_Y_coeffs_wrt_tau'][0]

sf_toe_X_coeffs = gait_params['sf_toe_X_coeffs_wrt_tau'][0]
sf_toe_Y_coeffs = gait_params['sf_toe_Y_coeffs_wrt_tau'][0]
nsf_toe_X_coeffs = gait_params['nsf_toe_X_coeffs_wrt_tau'][0]
nsf_toe_Y_coeffs = gait_params['nsf_toe_Y_coeffs_wrt_tau'][0]

torso_com_X_coeffs = gait_params['torso_com_X_coeffs_wrt_tau'][0]
torso_com_Y_coeffs = gait_params['torso_com_Y_coeffs_wrt_tau'][0]

com_X_coeffs = gait_params['com_X_coeffs_wrt_tau'][0]
com_Y_coeffs = gait_params['com_Y_coeffs_wrt_tau'][0]

def eval_poly(coeffs, x):
    """
    Evaluates polynomial of degree 4 at x
    """
    res = 0
    for i, c in enumerate(reversed(coeffs)):
        res += c * np.power(x, i)
    return res

cycle_length = 71

# nonzero_indices = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
#                    16, 17, 18, 19, 26, 27, 28, 29, 30, 31,
#                    32, 33, 34, 35, 36, 37]

nonzero_indices = [0, 6, 7, 8, 9, 10, 11, 18, 19, 26, 27, 28, 29, 30, 31,
                   32, 33, 34, 35]
x_indices = [18, 26, 28, 30, 32, 34]
left_stance_obs = []
right_stance_obs = []

# construct observation maps for LEFT foot being stance foot
for i in range(cycle_length):
    t = i * 0.01
    tau = eval_poly(tau_coeffs, t)
    obs = np.zeros(41)
    
    # 0: rotation of pelvis
    obs[0] = eval_poly(torso_coeffs, tau)
    
    # 1, 2: x, y position of pelvis
    # obs[1] = eval_poly(torso_com_X_coeffs, tau)
    # obs[2] = eval_poly(torso_com_Y_coeffs, tau)

    # 3, 4, 5: velocity (rotation, x, y) of pelvis    

    # 6-11: rotation of each ankle, knee, hip
    # order is: ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
    obs[6] = eval_poly(qd_coeffs[2], tau)
    obs[7] = eval_poly(qd_coeffs[1], tau)
    obs[8] = eval_poly(qd_coeffs[0], tau)
    obs[9] = eval_poly(qd_coeffs[3], tau)
    obs[10] = eval_poly(qd_coeffs[4], tau)
    obs[11] = eval_poly(qd_coeffs[5], tau)
    
    # 12-17: angular velocity of each ankle, knee, hip
    obs[12] = eval_poly(dot_qd_coeffs[2], tau)
    obs[13] = eval_poly(dot_qd_coeffs[1], tau)
    obs[14] = eval_poly(dot_qd_coeffs[0], tau)
    obs[15] = eval_poly(dot_qd_coeffs[3], tau)
    obs[16] = eval_poly(dot_qd_coeffs[4], tau)
    obs[17] = eval_poly(dot_qd_coeffs[5], tau)
    
    # 18, 19: x, y of COM
    obs[18] = eval_poly(com_X_coeffs, tau) - (-0.00388174517697 - -0.0697193532044)
    obs[19] = eval_poly(com_Y_coeffs, tau) - (0.998920554125 - 0.970765639028)

    # 20, 21: velocity of COM

    # 22, 23: position of head

    # 24, 25: position of pelvis

    # 26, 27: position of torso
    obs[26] = eval_poly(torso_com_X_coeffs, tau) - (-0.00162897940695 + 0.0965008489262)
    obs[27] = eval_poly(torso_com_Y_coeffs, tau) - (1.26664398555 - 0.996431048568)
    
    # 28, 29: position of left toe
    obs[28] = eval_poly(sf_toe_X_coeffs, tau) - (0.13 - 0.00798)
    obs[29] = eval_poly(sf_toe_Y_coeffs, tau) - 0.0274
    
    # 30, 31: position of right toe
    obs[30] = eval_poly(nsf_toe_X_coeffs, tau) - (0.13 - 0.00798)
    obs[31] = eval_poly(nsf_toe_Y_coeffs, tau) - 0.0274
    
    # 32, 33: position of left talus
    obs[32] = eval_poly(sf_talus_X_coeffs, tau) - 0.119683331742
    obs[33] = eval_poly(sf_talus_Y_coeffs, tau) - (0.0417232320769 - 0.0229523985286)
    
    # 34, 35: position of right talus
    obs[34] = eval_poly(nsf_talus_X_coeffs, tau) - 0.119683331742
    obs[35] = eval_poly(nsf_talus_Y_coeffs, tau) - (0.0417232320769 - 0.0229523985286)
    
    # 36, 37: strength of left and right psoas
    obs[36] = 1
    obs[37] = 1
    
    # 38, 39: distance of next obstacle 
    # 40: radius of obstacle

    left_stance_obs.append(obs)

# construct observation maps for RIGHT foot being stance foot
for i in range(cycle_length):
    t = i * 0.01
    tau = eval_poly(tau_coeffs, t)
    obs = np.zeros(41)
    
    # 0: rotation of pelvis
    obs[0] = eval_poly(torso_coeffs, tau)
    
    # 1, 2: x, y position of pelvis
    # obs[1] = eval_poly(torso_com_X_coeffs, tau)
    # obs[2] = eval_poly(torso_com_Y_coeffs, tau)

    # 3, 4, 5: velocity (rotation, x, y) of pelvis    

    # 6-11: rotation of each ankle, knee, hip
    # order is: ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
    obs[6] = eval_poly(qd_coeffs[3], tau) 
    obs[7] = eval_poly(qd_coeffs[4], tau)
    obs[8] = eval_poly(qd_coeffs[5], tau)
    obs[9] = eval_poly(qd_coeffs[2], tau)
    obs[10] = eval_poly(qd_coeffs[1], tau)
    obs[11] = eval_poly(qd_coeffs[0], tau)
    
    # 12-17: angular velocity of each ankle, knee, hip
    obs[12] = eval_poly(dot_qd_coeffs[3], tau)
    obs[13] = eval_poly(dot_qd_coeffs[4], tau)
    obs[14] = eval_poly(dot_qd_coeffs[5], tau)
    obs[15] = eval_poly(dot_qd_coeffs[2], tau)
    obs[16] = eval_poly(dot_qd_coeffs[1], tau)
    obs[17] = eval_poly(dot_qd_coeffs[0], tau)
    
    # 18, 19: x, y of COM
    obs[18] = eval_poly(com_X_coeffs, tau) - (-0.00388174517697 - -0.0697193532044)
    obs[19] = eval_poly(com_Y_coeffs, tau) - (0.998920554125 - 0.970765639028)
    
    # 20, 21: velocity of COM

    # 22, 23: position of head

    # 24, 25: position of pelvis

    # 26, 27: position of torso
    obs[26] = eval_poly(torso_com_X_coeffs, tau) - (-0.00162897940695 + 0.0965008489262)
    obs[27] = eval_poly(torso_com_Y_coeffs, tau) - (1.26664398555 - 0.996431048568)
    
    # 28, 29: position of left toe
    obs[28] = eval_poly(nsf_toe_X_coeffs, tau) - (0.13 - 0.00798)
    obs[29] = eval_poly(nsf_toe_Y_coeffs, tau) - 0.0274
    
    # 30, 31: position of right toe
    obs[30] = eval_poly(sf_toe_X_coeffs, tau) - (0.13 - 0.00798)
    obs[31] = eval_poly(sf_toe_Y_coeffs, tau) - 0.0274
    
    # 32, 33: position of left talus
    obs[32] = eval_poly(nsf_talus_X_coeffs, tau) - 0.119683331742
    obs[33] = eval_poly(nsf_talus_Y_coeffs, tau) - (0.0417232320769 - 0.0229523985286)
    
    # 34, 35: position of right talus
    obs[34] = eval_poly(sf_talus_X_coeffs, tau) - 0.119683331742
    obs[35] = eval_poly(sf_talus_Y_coeffs, tau) - (0.0417232320769 - 0.0229523985286)
    
    # 36, 37: strength of left and right psoas
    obs[36] = 1
    obs[37] = 1
    
    # 38, 39: distance of next obstacle 
    # 40: radius of obstacle

    right_stance_obs.append(obs)

# Load walking environment
env = RunEnv(args.visualize)
env.reset()
env.set_imitation(left_stance_obs, right_stance_obs, cycle_length, nonzero_indices, x_indices)

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

    
# ts1 = []
# ts2 = []
# training_set = []
# for i in range(1, 101):
#     t = 0.005 * i
#     tau = eval_poly(tau_coeffs, t)
#     obs = np.zeros(env.observation_space.shape[0])
#     # rotation of pelvis/torso
#     obs[0] = eval_poly(torso_coeffs, t)
#     # angular velocity of each ankle, knee, hip
#     for j in range(6):
#         obs[6+j] = eval_poly(qd_coeffs[j], t)
#         obs[12+j] = eval_poly(dot_qd_coeffs[j], t)
#     training_set.append(np.concatenate([np.random.random(size=nb_actions), obs]))
#     for j in range(100):
#         ts1.append(0.25 * np.random.random(size=nb_actions))
#         ts2.append(obs)

# training_set = np.array(training_set)
# ts1 = np.array(ts1)
# ts2 = np.array(ts2)

# labels = 2 * np.ones(100 * len(training_set))

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
#print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
#print(critic.summary())


# ts2 = ts2.reshape([10000, 1, 41])
# critic.compile(Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])
# critic.fit([ts1, ts2], labels, epochs=10)

# Set up the agent for training
memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.15, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    #agent.load_weights(args.model)
    agent.fit(env, nb_steps=nallsteps, action_repetition=1, visualize=False, verbose=2, nb_max_episode_steps=env.timestep_limit, log_interval=10000)
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

# If TEST and TOKEN, submit to crowdAI
if not args.train and args.token:
    agent.load_weights(args.model)
    # Settings
    remote_base = 'http://grader.crowdai.org:1729'
    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token)

    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    while True:
        v = np.array(observation).reshape((env.observation_space.shape[0]))
        action = agent.forward(v)
        [observation, reward, done, info] = client.env_step(action.tolist())
        if done:
            observation = client.env_reset()
            if not observation:
                break

    client.submit()

# If TEST and no TOKEN, run some test experiments
if not args.train and not args.token:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
