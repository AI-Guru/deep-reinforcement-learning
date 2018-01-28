""" This is a prototype for Deep Reinforcement Learning. """

from vizdoom import *
import random
import time
import keras
from keras import models
from keras import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt

# TODO use this: https://keon.io/deep-q-learning/

# Creates the Doom-game.
game = DoomGame()
game.load_config("vizdoom/scenarios/basic.cfg")
game.set_screen_format(ScreenFormat.GRAY8)
game.init()

# These are the available actions.
action_none = [0, 0, 0]
action_shoot = [0, 0, 1]
action_left = [1, 0, 0]
action_right = [0, 1, 0]
actions = [action_shoot, action_left, action_right]

# Model parameters.
model_type = "dense"
#model_type = "conv3d"
target_shape = (45, 30, 3)
memory_size = 10
action_size = 3

# Training parameters.
epochs = 20 # Number of epochs to train.
learning_steps_per_epoch = 2000
frame_skip = 5 # Number of frames to skip during training.

class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def main(run=False):
    """ This is just the main method. """
    if run == True:
        run()
    else:
        train()


def run():
    """ Runs the trained model. """
    # TODO
    raise Exception("Implement!")


def train():
    """ Trains the model. """

    # Create the model.
    model = create_model(target_shape, memory_size, len(action_none))

    # Ensure that the memory is initialzed.
    global memory
    memory = initialize_memory(target_shape, memory_size)

    # Do the training.
    for epoch in range(epochs):
        game.new_episode()

        for _ in range(learning_steps_per_epoch):
            if game.is_episode_finished():
                game.new_episode()
            else:
                train_in_game(game, model, epoch)


def train_in_game(game, model, epoch):
    state = game.get_state()

    misc = state.game_variables

    global memory
    screen_buffer = state.screen_buffer
    screen_buffer = transform_screen_buffer(screen_buffer, target_shape)
    memory.append(screen_buffer)
    if len(memory) > memory_size:
        memory.pop()


    #frame_count = 0

    # Update the memory with the last screen.
    #if frame_count == 0:
    #    screen_buffer = transform_screen_buffer(screen_buffer, target_shape)
    #    memory.append(screen_buffer)
    #    if len(memory) > memory_size:
    #        memory.pop()
    #frame_count = (frame_count + 1) % frame_skip

    # Transform into something that keras can train on.
    memory_array = np.array(memory)
    memory_array = np.expand_dims(memory_array, axis = 0)

    # Get an action. Either a random one, or a predicted one.
    if random.random() < exploration_rate(epoch):
        action = random.choice(actions)
    else:
        action = predict_action(model, memory_array)

    # Perform the action and gather result.
    reward = game.make_action(action)

    # Train the model
    action_array = np.array(action)
    action_array = np.expand_dims(action_array, axis = 0)
    reward_array = np.array([reward])
    reward_array = np.expand_dims(reward_array, axis = 0)
    train_model(model, memory_array, action_array, reward_array)

    print("Result:", game.get_total_reward())
    # TODO save model.

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def initialize_memory(target_shape, memory_size):
    """ Fills the memory with black images. """
    memory = []
    for _ in range(memory_size):
        element = np.zeros(target_shape)
        memory.append(element)
    return memory


def create_model(target_shape, memory_size, action_size):
    """ Creates the model for training. """

    input_memory = keras.layers.Input(shape=(memory_size,) + target_shape, name="input_memory")
    input1 = input_memory

    # The second input is the encoded instrument.
    #input_action = keras.layers.Input(shape=(action_size,), name="input_action")
    #input2 = input_action

    # TODO find the proper architecture.
    if model_type == "dense":
        input1 = keras.layers.Flatten()(input1)
        x = keras.layers.concatenate([input1, input2])
        x = keras.layers.Dense(128)(x)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.Dense(8)(x)
    elif model_type == "conv2d":
        input1 = keras.layers.Conv2D(8, (6, 6), (3, 3))(input1)
        input1 = keras.layers.Conv2D(8, (3, 3), (2, 2))(input1)
        input1 = keras.layers.Flatten()(input1)
        x = keras.layers.concatenate([input1, input2])
        x = keras.layers.Dense(128, activation="relu", name="output")(x)
    elif model_type == "conv3d":
        input1 = keras.layers.Conv3D(32, (3, 3, 3))(input1)
        input1 = keras.layers.MaxPooling3D((2, 2, 2))(input1)
        input1 = keras.layers.Flatten()(input1)
        x = keras.layers.concatenate([input1, input2])
    else:
        raise Exception("Unknown model type:", model_type)

    # Properly create the model and return.
    output = keras.layers.Dense(action_size, activation="sigmoid", name="output")(x)
    model = keras.models.Model(inputs=[input_memory, input_action], outputs=[output])
    model.summary()

    # Compiles the model.
    # TODO is this right?
    model.compile(
        loss = "mse",
        optimizer = "rmsprop",
        metrics=["accuracy"]
    )

    return model


def transform_screen_buffer(screen_buffer, target_shape):
    """ Transforms the screen buffer for the neural network. """

    # If it is grayscale, add another dimension.
    if screen_buffer.ndim == 2:
        screen_buffer = np.expand_dims(screen_buffer, axis=2)
    # If it is RGB, swap the axes.
    elif screen_buffer.ndom == 3:
        screen_buffer = np.swapaxes(screen_buffer, 0, 2)
        screen_buffer = np.swapaxes(screen_buffer, 0, 1)

    screen_buffer = cv2.resize(screen_buffer, (target_shape[0], target_shape[1]))
    screen_buffer = screen_buffer.astype("float32") / 255.0
    return screen_buffer


def predict_action(model, memory_array):
    """ Predicts an action using the current model. """

    max_value = 0.0
    max_action = [0, 0, 0]
    for action in actions:
        action_array = np.array(action)
        action_array = np.expand_dims(action_array, axis=0)
        predicted_reward = model.predict([memory_array, action_array])[0]
        if predicted_reward > max_value:
            max_value = predicted_reward
            max_action = action

    return max_action


def exploration_rate(epoch):
    start_eps = 1.0
    end_eps = 0.1
    const_eps_epochs = 0.1 * epochs  # 10% of learning time
    eps_decay_epochs = 0.6 * epochs  # 60% of learning time

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
                           (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps

def train_model(model, memory_array, action_array, reward_array):
    """ Trains the model on the current memory, an action, and a reward. """

    model_input = {
        "input_memory": memory_array,
        "input_action": action_array
        }
    model_output = {
        "output": reward_array
    }
    history = model.fit(
        model_input, model_output,
        epochs=1,
        batch_size=1,
        verbose=0)


if __name__ == "__main__":
    main()
