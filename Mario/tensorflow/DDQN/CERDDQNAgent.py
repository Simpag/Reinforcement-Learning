import os
import random
import gym
from collections import deque
import time
import numpy as np

from DDQN.ModifiedTensorBoard import ModifiedTensorBoard

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import Huber # basically smooth L1 loss
from keras import backend as K

import tensorflow as tf

import gc


class CERDDQNAgent():
    def __init__(self, ENV: gym.Env, DISCOUNT: float, LEARNING_RATE: int, TARGET_MODEL_UPDATE_CYCLE: int, REPLAY_MEMORY_SIZE: int, MINIBATCH_SIZE: int, MIN_REPLAY_MEMORY_SIZE: int, MODEL_NAME: str, MODEL_TO_LOAD = None, LOG_DIR = None) -> None:
        # DQ variables
        self.DISCOUNT = DISCOUNT
        self._lr = LEARNING_RATE

        # An array with last n steps for training, to batch train
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.newest_transition = None
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE

        # Custom tensorboard object
        if LOG_DIR is None:
            self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        else:
            self.tensorboard = ModifiedTensorBoard(log_dir=LOG_DIR)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Save number of actions
        self.num_actions = ENV.action_space.n

        # Save obs space shape
        self.observation_space = ENV.observation_space.shape
        print(self.observation_space)

        # Main model
        if MODEL_TO_LOAD is None:
            self.model = self.create_model()
            self.target_model = self.create_model() # Want the model that we query for future Q values to be more stable than the model that we're actively fitting every single step
        else:
            self.model = self.load(MODEL_TO_LOAD)
            self.target_model = self.load(MODEL_TO_LOAD) # Want the model that we query for future Q values to be more stable than the model that we're actively fitting every single step

        self.target_model.trainable = False

        # Optimizer
        self.optimizer = Adam(learning_rate=self._lr)
        
        # Loss function
        self.loss_function = Huber()

        # Target network | Used for predicting so we dont "overfit", initially the network will fluctuate a lot
        self.target_model.set_weights(self.model.get_weights())
        self.target_model_update_cycle = TARGET_MODEL_UPDATE_CYCLE # How many episodes before it updates
    
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=8, strides=4, input_shape=self.observation_space))  # env.observation_space a nxn RGB image.
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=4, strides=2))
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=3, strides=1))
        model.add(Activation('relu'))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        model.add(Dense(512))
        model.add(Activation('relu'))

        model.add(Dense(self.num_actions))  # ACTION_SPACE_SIZE = how many choices
        model.add(Activation('linear'))
        #model.compile(loss="mse", optimizer=Adam(learning_rate=self._lr), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    def update_replay_memory(self, transition: tuple):
        # transition = (current_state, action, reward, new_state, done)
        self.newest_transition = transition

    # Add the newest transition to the replay memory
    def _update_replay_memory(self):
        self.replay_memory.append(self.newest_transition)
        self.newest_transition = None


    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            if self.newest_transition is not None:
                self._update_replay_memory()
            return
            
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE-1)
        # Add the newest transition to the minibatch according to CER
        if self.newest_transition is not None:
            minibatch.append(self.newest_transition)
            # Append the newest transition to replay memory
            self._update_replay_memory()

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([np.array(transition[0]) for transition in minibatch])/255

        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array([np.array(transition[3]) for transition in minibatch])/255
        future_qs_list = self.target_model(new_current_states, training=False)
        #future_qs_list = self.target_model.predict(new_current_states)

        # Get reward sample
        rewards = [transition[2] for transition in minibatch]

        # Get action sample
        actions = [transition[1] for transition in minibatch]

        # Get all the done transitions
        dones = tf.convert_to_tensor([float(transition[4]) for transition in minibatch])

        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards + self.DISCOUNT * tf.reduce_max(
                future_qs_list, axis=1
            )
        
        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - dones) - dones

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(actions, self.num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            current_qs_list = self.model(current_states)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(current_qs_list, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.target_model_update_cycle:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        #gc.collect()        # without this I get mad memory leak
        #K.clear_session()   # without this I get mad memory leak

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