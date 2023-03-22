import random
import time
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from multiprocessing import Pool

from keras import backend as K
import gc

import gym
from gym import logger
logger.set_level(logger.ERROR)
import gym_snake


# To enable multiple processes or something idk, just run it before the script
# export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

def main():
    from DQNAgent import DQNAgent

    # Model settings
    MODEL_NAME = f"8a_final"
    MODEL_TO_LOAD = None                # Load model from file, (None = wont load)
    TARGET_MODEL_UPDATE_CYCLE = 500     # Number of terminal states before updating target model
    REPLAY_MEMORY_SIZE = 25_000         # How big the batch size should be
    MIN_REPLAY_MEMORY_SIZE = 1_000      # Number of steps recorded before training starts

    # Training settings
    STARTING_EPISODE = 1                # Which episode to start from (should be 1 unless continued training on a model)
    EPISODES = 15_000                   # Total training episodes
    MINIBATCH_SIZE = 32              # How many steps to use for training

    #  Stats settings
    MIN_REWARD = 50                     # Save model that reaches min avg reward
    AGGREGATE_STATS_EVERY = 50          # When to record data to plot how it performs
    SHOW_PREVIEW = False                # Show preview of agent playing

    # DQ-settings
    DISCOUNT = 0.99                     # gamma (discount factor)
    LEARNING_RATE = 0.00001 #0.001

    # Exploration settings
    epsilon = 1                         
    EPSILON_DECAY = 0.9999 #0.99975 #0.95
    MIN_EPSILON = 0.001

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)

    # For stats
    ep_rewards = [-1,] 

    env = gym.make("Snake-16x16-8a-v0")

    agent = DQNAgent(env, DISCOUNT, LEARNING_RATE, TARGET_MODEL_UPDATE_CYCLE, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, MODEL_NAME, MODEL_TO_LOAD)

    for episode in tqdm(range(STARTING_EPISODE, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        steps_without_reward = 0
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset() #env.reset(seed=ENV_SEED) # not supported by snake env ig

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            if np.random.random() > epsilon:
                # Get action from DQN
                action = agent.get_action(current_state)
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, truncated = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1


        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                agent.save(f'{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if not episode % 1000:
            agent.save(f'{MODEL_NAME}_episode_{episode}_{epsilon}epsilon_{int(time.time())}.model')


        # Decay epsilon, exponential
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    return 0


if __name__ == "__main__":
    # look for batch size (compare with CER)
    main()
    
    