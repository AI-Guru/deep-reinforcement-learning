from vizdoom import *
import random
import time
import keras
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

game = DoomGame()
game.load_config("vizdoom/scenarios/basic.cfg")
game.set_screen_format(ScreenFormat.GRAY8)
game.init()

epochs = 20 # Number of epochs to train.
learning_steps_per_epoch = 2000

screen_shape = (30, 45, 1)

action_none = [0, 0, 0]
action_shoot = [0, 0, 1]
action_left = [1, 0, 0]
action_right = [0, 1, 0]
actions = [action_shoot, action_left, action_right]
action_length = 3

#model_type = "dense"
model_type = "conv2d"

class DQNAgent:

    def __init__(self, model_type, screen_shape, action_length, actions):
        self.screen_shape = screen_shape
        self.action_length = action_length
        self.actions = actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(model_type)


    def _build_model(self, model_type):
        if model_type == "dense":
            model = models.Sequential()
            model.add(layers.Flatten(input_shape=self.screen_shape))
            model.add(layers.Dense(24, activation='relu'))
            model.add(layers.Dense(24, activation='relu'))
            model.add(layers.Dense(self.action_length, activation='linear'))
        elif model_type == "conv2d":
            model = models.Sequential()
            model.add(layers.Conv2D(8, (6, 6), strides=(3, 3), input_shape=self.screen_shape))
            model.add(layers.Conv2D(8, (3, 3), strides=(2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(125, activation='relu'))
            model.add(layers.Dense(self.action_length, activation='linear'))

        model.compile(loss='mse',
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model


    def remember(self, screen, action, reward, next_screen, done):
        assert screen.shape == self.screen_shape
        assert next_screen.shape == self.screen_shape
        #print("Remember")
        #print("state", state)
        #print("action", action)
        #print("reward", reward)
        #print("next_state", next_state)+
        #print("  state", state.shape)
        #print("  next_state", next_state.shape)

        self.memory.append((screen, action, reward, next_screen, done))


    def act(self, screen):
        if np.random.rand() <= self.epsilon: # TODO use an exploration rate here
            return random.choice(self.actions)
        screen = np.expand_dims(screen, axis=0)
        act_values = self.model.predict(screen)
        max_index = np.argmax(act_values[0])
        return self.actions[max_index]


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for screen, action, reward, next_screen, done in minibatch:
            screen = np.expand_dims(screen, axis=0)
            next_screen = np.expand_dims(next_screen, axis=0)

            target = reward
            if not done:
                #print(next_state)
                prediction = self.model.predict(next_screen)[0]
                target = (reward + self.gamma *
                          np.amax(prediction))
            target_f = self.model.predict(screen)
            target_f[0][action] = target
            self.model.fit(screen, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)


def main():

    agent = DQNAgent(model_type, screen_shape, action_length, actions)

    done = False
    batch_size = 32

    # Do the training.
    done = True
    for epoch in range(epochs):
        game.new_episode()

        for _ in range(learning_steps_per_epoch):
            if done == True:
                done = False
                game.new_episode()
                state = game.get_state()
                screen = state.screen_buffer
                screen = transform_screen_buffer(screen, screen_shape)
                continue

            action = agent.act(screen)
            reward = game.make_action(action)
            next_state = game.get_state()
            if game.is_episode_finished():
                done = True
                continue

            next_screen = next_state.screen_buffer

            next_screen = transform_screen_buffer(next_screen, screen_shape)
            agent.remember(screen, action, reward, next_screen, done)
            screen = next_screen
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)



def transform_screen_buffer(screen_buffer, target_shape):
    """ Transforms the screen buffer for the neural network. """

    # If it is RGB, swap the axes.
    if screen_buffer.ndim == 3:
        screen_buffer = np.swapaxes(screen_buffer, 0, 2)
        screen_buffer = np.swapaxes(screen_buffer, 0, 1)

    # Resize.
    screen_buffer = cv2.resize(screen_buffer, (target_shape[1], target_shape[0]))

    # If it is grayscale, add another dimension.
    if screen_buffer.ndim == 2:
        screen_buffer = np.expand_dims(screen_buffer, axis=2)

    screen_buffer = screen_buffer.astype("float32") / 255.0

    return screen_buffer


if __name__ == "__main__":
    main()
