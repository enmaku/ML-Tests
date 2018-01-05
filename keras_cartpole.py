import numpy as np
import gym
import pathlib

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy

app_name = 'keras_cartpole'
fit = True
test = True
visualize_fit = False
visualize_test = True

gym.envs.register(id='CartPoleLong-v0', entry_point='gym.envs.classic_control:CartPoleEnv', max_episode_steps=20000)
env = gym.make('CartPoleLong-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model_folder = './models/' + app_name + '/'
model_file = model_folder + app_name + '.h5'

try:
    # Load the model if it already exists.
    print('Loading existing model...')
    model = load_model(model_file)
    print('Model loaded.')
except OSError:
    # Build it from scratch if it doesn't.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64, activation='relu', use_bias=True, name='dense1'))
    model.add(Dense(64, activation='relu', use_bias=True, name='dense2'))
    model.add(Dense(64, activation='relu', use_bias=True, name='dense3'))
    model.add(Dense(nb_actions, activation='linear', name='readout'))
    print(model.summary())

policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

if fit:
    sarsa.fit(env, nb_steps=50000, visualize=visualize_fit, verbose=2)
    pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
    model.save(model_file)

if test:
    sarsa.test(env, nb_episodes=5, visualize=visualize_test)
