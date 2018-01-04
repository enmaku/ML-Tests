import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

gym.envs.register(id='CartPoleLong-v0', entry_point='gym.envs.classic_control:CartPoleEnv', max_episode_steps=2000)
env = gym.make('CartPoleLong-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(5,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=25000, window_length=5)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=25000, visualize=False, verbose=2)

dqn.test(env, nb_episodes=25, visualize=True)
