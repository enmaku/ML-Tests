import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, env, model, memory_length=2000, batch_size=32, gamma=0.85, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, tau=0.125, visualize=False):
        self.env = env
        self.model = model
        self.memory = deque(maxlen=memory_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.visualize = visualize
        self.action_model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        return self.model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if self.visualize:
            self.env.render()
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.action_model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future * self.gamma
            self.action_model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.action_model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.action_model.save(fn)
