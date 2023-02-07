import random
import time
from DQNAgent import DQNAgent
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import gym
import gym_snake

# export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

def main():
    # Model settings
    TARGET_MODEL_UPDATE_CYCLE = 5   # Number of terminal states before updating target model
    REPLAY_MEMORY_SIZE = 50_000     # How big the batch size should be
    MIN_REPLAY_MEMORY_SIZE = 1_000  # Number of steps recorded before training starts
    MODEL_NAME = "Snake_16x16"

    # Training settings
    EPISODES = 20_000               # Total training episodes
    MINIBATCH_SIZE = 64             # How many steps to use for training

    #  Stats settings
    MIN_REWARD = 1                  # Save model that reaches min reward
    AGGREGATE_STATS_EVERY = 50      # When to record data to plot how it performs
    SHOW_PREVIEW = False            # Show preview of agent playing

    # DQ-settings
    DISCOUNT = 0.99                 # gamma (discount factor)
    LEARNING_RATE = 0.001

    # Exploration settings
    epsilon = 1                     # Not a constant, going to be decayed
    EPSILON_DECAY = 0.95 #0.99975
    MIN_EPSILON = 0.001

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)

    # For stats
    ep_rewards = [0] 

    #env = gym.make("Snake-16x16-big-apple-reward-v0") 
    env = gym.make("Snake-16x16-v0") 

    agent = DQNAgent(env, DISCOUNT, LEARNING_RATE, TARGET_MODEL_UPDATE_CYCLE, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, MODEL_NAME)

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset() #env.reset(seed=ENV_SEED) # not supported by snake env ig

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
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

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.save(f'{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon, exponential
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

if __name__ == "__main__":
    main()