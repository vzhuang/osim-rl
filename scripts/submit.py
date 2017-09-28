import opensim as osim
from osim.http.client import Client
from osim.env import *
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, BatchNormalization
from keras.optimizers import Adam
import numpy as np
import argparse


from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


# Settings
remote_base = 'http://grader.crowdai.org:1729'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
parser.add_argument("--model", dest='model', action='store', default='example.h5f')
args = parser.parse_args()

env = RunEnv(visualize=False)
client = Client(remote_base)

# load model

nb_actions = env.action_space.shape[0]

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(128))
actor.add(Activation('relu'))
actor.add(BatchNormalization())
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(BatchNormalization())
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

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
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.5, mu=0., sigma=.5, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
agent.compile(Adam(lr=0.001, clipnorm=1.),metrics=['mae'])
# load pretrained weights
agent.load_weights(args.model)


# Create environment
observation = client.env_create(args.token)
print observation

# Run a single step
#
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    v = np.array(observation).reshape((-1,1,env.observation_space.shape[0]))
    [observation, reward, done, info] = client.env_step(actor.predict(v)[0].tolist())
    # print(observation)
    #print observation, reward
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
