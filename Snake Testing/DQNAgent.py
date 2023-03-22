import os
import random
import gym
from collections import deque
import time
import numpy as np

from ModifiedTensorBoard import ModifiedTensorBoard

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

import gc

class DQNAgent():
    def __init__(self, ENV: gym.Env, DISCOUNT: float, LEARNING_RATE: int, TARGET_MODEL_UPDATE_CYCLE: int, REPLAY_MEMORY_SIZE: int, MINIBATCH_SIZE: int, MIN_REPLAY_MEMORY_SIZE: int, MODEL_NAME: str, MODEL_TO_LOAD = None, LOG_DIR = None, HIDDEN_LAYERS = 128) -> None:
        # DQ variables
        self.DISCOUNT = DISCOUNT
        self._lr = LEARNING_RATE

        # An array with last n steps for training, to batch train
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE

        self.HIDDEN_LAYERS = HIDDEN_LAYERS

        # Custom tensorboard object
        if LOG_DIR is None:
            self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        else:
            self.tensorboard = ModifiedTensorBoard(log_dir=LOG_DIR)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Main model
        if MODEL_TO_LOAD is None:
            self.model = self.create_model(ENV)
            self.target_model = self.create_model(ENV) # Want the model that we query for future Q values to be more stable than the model that we're actively fitting every single step
        else:
            self.model = self.load(MODEL_TO_LOAD)
            self.target_model = self.load(MODEL_TO_LOAD) # Want the model that we query for future Q values to be more stable than the model that we're actively fitting every single step

        # Target network | Used for predicting so we dont "overfit", initially the network will fluctuate a lot
        self.target_model.set_weights(self.model.get_weights())
        self.target_model_update_cycle = TARGET_MODEL_UPDATE_CYCLE # How many episodes before it updates

        self._clear_memory_at = 10
        self._clear_memory_counter = 1
    """
    def create_model(self, env: gym.Env):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.observation_space.shape))  # env.observation_space a nxn RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.action_space.n, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self._lr), metrics=['accuracy'])
        return model

    def create_model(self, env: gym.Env):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape=env.observation_space.shape))  # env.observation_space a nxn RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.action_space.n, activation='linear'))  # ACTION_SPACE_SIZE = how many choices
        model.compile(loss="mse", optimizer=Adam(learning_rate=self._lr), metrics=['accuracy'])
        return model"""

    def create_model(self, env: gym.Env):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=8, strides=4, input_shape=env.observation_space.shape, activation='relu'))  # env.observation_space a nxn RGB image.

        model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'))

        model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        model.add(Dense(self.HIDDEN_LAYERS, activation='relu'))

        model.add(Dense(env.action_space.n, activation='linear'))  # ACTION_SPACE_SIZE = how many choices
        model.compile(loss="mse", optimizer=Adam(learning_rate=self._lr), metrics=['accuracy'])
        return model # maybe try softmax activation at the last connected layer

    # Adds step's data to a memory replay array
    def update_replay_memory(self, transition: tuple):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
            
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model(current_states).numpy()
        #current_qs_list = self.model.predict(current_states, verbose=0)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model(new_current_states, training=False).numpy()
        #future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = [] # left side of network
        y = [] # right side of network

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q # why not use the (1 - learning rate) * old Q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
            self._clear_memory_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.target_model_update_cycle:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        if self._clear_memory_counter > self._clear_memory_at:
            gc.collect()        # without this I get mad memory leak
            K.clear_session()   # without this I get mad memory leak
            self._clear_memory_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_action(self, state):
        # TODO understand this [0] stuff (probably because reshape(-1,...) is an unknown length)
        #return self.model.predict(np.array(state).reshape(-1, *state.shape)/255, verbose=0, use_multiprocessing=True)[0]
        #state = np.array(state).reshape(-1, *state.shape)/255
        state_tensor = tf.convert_to_tensor(np.array(state)/255)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        return tf.argmax(action_probs[0]).numpy()

    def save(self, filename):
        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')
        save_model(self.model, 'models/' + filename)
        print(f"saved model at: {'models/' + filename}")
        gc.collect()
        K.clear_session()

    def load(self, filename):
        print(f"Loaded model: {filename}")
        return load_model(filename)