import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from customlearn.algorithms import DQN

env = gym.make("MountainCar-v0")

trials = 1000
trial_len = 500

model = Sequential()
model.add(Dense(24, input_dim=env.observation_space.shape[0], activation="relu"))
model.add(Dense(48, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(env.action_space.n))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.005))

# updateTargetNetwork = 1000
dqn_agent = DQN(env=env, model=model, visualize=True)
steps = []
max_reward = 0
for trial in range(trials):
    cur_state = env.reset().reshape(1, 2)
    for step in range(trial_len):
        action = dqn_agent.act(cur_state)
        new_state, reward, done, _ = env.step(action)

        # reward = reward if not done else -20
        new_state = new_state.reshape(1, 2)
        dqn_agent.remember(cur_state, action, reward, new_state, done)

        dqn_agent.replay()  # internally iterates default (prediction) model
        dqn_agent.target_train()  # iterates target model

        cur_state = new_state
        if done:
            break
    if step >= 199:
        print("Failed to complete in trial {}.".format(trial))
    else:
        print("Completed in {} trials".format(trial))
        break
