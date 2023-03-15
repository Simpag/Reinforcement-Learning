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

def main(folder, lr, ed, bs, tu, it, tqdm_name, rm = 25_000):
    from DQNAgent import DQNAgent

    # Model settings
    MODEL_NAME = f"{folder}/16x16_8a_bs{bs}_lr{lr}_ed{ed}_tu{tu}"
    MODEL_TO_LOAD = None                # Load model from file, (None = wont load)
    TARGET_MODEL_UPDATE_CYCLE = tu #10      # Number of terminal states before updating target model
    REPLAY_MEMORY_SIZE = rm #25_000         # How big the batch size should be
    MIN_REPLAY_MEMORY_SIZE = min(rm, 1_000) # Number of steps recorded before training starts

    # Training settings
    STARTING_EPISODE = 1                # Which episode to start from (should be 1 unless continued training on a model)
    EPISODES = 15_000                   # Total training episodes
    MINIBATCH_SIZE = bs#32              # How many steps to use for training

    #  Stats settings
    MIN_REWARD = 50                     # Save model that reaches min avg reward
    AGGREGATE_STATS_EVERY = 50          # When to record data to plot how it performs
    SHOW_PREVIEW = False                # Show preview of agent playing

    # DQ-settings
    DISCOUNT = 0.99                     # gamma (discount factor)
    LEARNING_RATE = lr #0.001

    # Exploration settings
    epsilon = 1                         
    EPSILON_DECAY = ed #0.9999 #0.99975 #0.95
    MIN_EPSILON = 0.001

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)

    # For stats
    ep_rewards = [-1,] 

    #env = gym.make("Snake-16x16-big-apple-reward-v0") 
    #env = gym.make("Snake-16x16-v0")
    #env = gym.make("Snake-16x16-heatmap-v0")
    #env = gym.make("Snake-16x16-heatmap-big-reward-v0")
    #env = gym.make("Snake-16x16-heatmap-big-reward-5-apples-v0")
    #env = gym.make("Snake-16x16-4a-v0")
    env = gym.make("Snake-16x16-8a-v0")

    agent = DQNAgent(env, DISCOUNT, LEARNING_RATE, TARGET_MODEL_UPDATE_CYCLE, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, MODEL_NAME, MODEL_TO_LOAD)

    for episode in tqdm(range(STARTING_EPISODE, EPISODES + 1), ascii=True, unit='episodes', position=it, desc=tqdm_name):
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

def main2(folder, lr, ed, bs, tu, it, tqdm_name, rm = 25_000):
    # USING CER agent
    from CERDQNAgent import CERDQNAgent

    # Model settings
    MODEL_NAME = f"{folder}/16x16_8a_bs{bs}_lr{lr}_ed{ed}_tu{tu}"
    MODEL_TO_LOAD = None                # Load model from file, (None = wont load)
    TARGET_MODEL_UPDATE_CYCLE = tu #10      # Number of terminal states before updating target model
    REPLAY_MEMORY_SIZE = rm #25_000         # How big the batch size should be
    MIN_REPLAY_MEMORY_SIZE = min(rm, 1_000) # Number of steps recorded before training starts

    # Training settings
    STARTING_EPISODE = 1                # Which episode to start from (should be 1 unless continued training on a model)
    EPISODES = 15_000                   # Total training episodes
    MINIBATCH_SIZE = bs#32              # How many steps to use for training

    #  Stats settings
    MIN_REWARD = 50                     # Save model that reaches min avg reward
    AGGREGATE_STATS_EVERY = 50          # When to record data to plot how it performs
    SHOW_PREVIEW = False                # Show preview of agent playing

    # DQ-settings
    DISCOUNT = 0.99                     # gamma (discount factor)
    LEARNING_RATE = lr #0.001

    # Exploration settings
    epsilon = 1                         
    EPSILON_DECAY = ed #0.9999 #0.99975 #0.95
    MIN_EPSILON = 0.001

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)

    # For stats
    ep_rewards = [-1,] 

    env = gym.make("Snake-16x16-8a-v0")

    agent = CERDQNAgent(env, DISCOUNT, LEARNING_RATE, TARGET_MODEL_UPDATE_CYCLE, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, MODEL_NAME, MODEL_TO_LOAD)

    for episode in tqdm(range(STARTING_EPISODE, EPISODES + 1), ascii=True, unit='episodes', position=it, desc=tqdm_name):
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


def flr(lr, it):
    if not os.path.isdir(f'models/learning_rate_test{version}/8a_lr{lr}'):
        os.makedirs(f'models/learning_rate_test{version}/8a_lr{lr}')
    main(folder=f"learning_rate_test{version}/8a_lr{lr}", lr=lr, ed=0.9995, bs=32, tu=10, it=it, tqdm_name=f'lr_{lr}')
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def fed(ed, it):
    if not os.path.isdir(f'models/epsilon_decay_test{version}/8a_ed{ed}'):
        os.makedirs(f'models/epsilon_decay_test{version}/8a_ed{ed}')
    main(folder=f"epsilon_decay_test{version}/8a_ed{ed}", lr=0.001, ed=ed, bs=32, tu=10, it=it, tqdm_name=f'ed_{ed}')
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def fbs(batch_size, it):
    if not os.path.isdir(f'models/batch_size_test{version}/8a_bs{batch_size}'):
        os.makedirs(f'models/batch_size_test{version}/8a_bs{batch_size}')
    main(folder=f"batch_size_test{version}/8a_bs{batch_size}", lr=0.001, ed=0.9995, bs=batch_size, tu=10, it=it, tqdm_name=f'bs_{batch_size}')
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def fbs2(batch_size, it):
    if not os.path.isdir(f'models/cer_batch_size_test{version}/8a_bs{batch_size}'):
        os.makedirs(f'models/cer_batch_size_test{version}/8a_bs{batch_size}')
    main2(folder=f"cer_batch_size_test{version}/8a_bs{batch_size}", lr=0.001, ed=0.9995, bs=batch_size, tu=10, it=it, tqdm_name=f'cer_bs_{batch_size}')
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def ftu(target_update, it):
    if not os.path.isdir(f'models/target_update_test{version}/8a_tu{target_update}'):
        os.makedirs(f'models/target_update_test{version}/8a_tu{target_update}')
    main(folder=f"target_update_test{version}/8a_tu{target_update}", lr=0.001, ed=0.9995, bs=32, tu=target_update, it=it, tqdm_name=f'tu_{target_update}')
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def frm(replay_memory, it):
    if not os.path.isdir(f'models/replay_memory_test{version}/8a_rm{replay_memory}'):
        os.makedirs(f'models/replay_memory_test{version}/8a_rm{replay_memory}')
    main(folder=f"replay_memory_test{version}/8a_rm{replay_memory}", lr=0.001, ed=0.9995, bs=32, tu=10, it=it, tqdm_name=f'rm_{replay_memory}', rm=replay_memory)
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def frm2(replay_memory, it):
    if not os.path.isdir(f'models/cer_replay_memory_test{version}/8a_rm{replay_memory}'):
        os.makedirs(f'models/cer_replay_memory_test{version}/8a_rm{replay_memory}')
    main2(folder=f"cer_replay_memory_test{version}/8a_rm{replay_memory}", lr=0.001, ed=0.9995, bs=32, tu=10, it=it, tqdm_name=f'cer_rm_{replay_memory}', rm=replay_memory)
    gc.collect()        # without this I get mad memory leak
    K.clear_session()   # without this I get mad memory leak

def f(i):
    """if i % 24 < 4:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(devices, 'GPU')
    else:
        tf.config.set_visible_devices([], 'GPU') # only run 4 on gpu
    """
    if i >= 4:
        tf.config.set_visible_devices([], 'GPU') # only run 4 on gpu
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    if i < len(learning_rates):
        flr(learning_rates[i], it=i)
        return
    else:
        i -= len(learning_rates)

    if i < len(epsilon_decays):
        fed(epsilon_decays[i], it=i+len(learning_rates))
        return
    else:
        i -= len(epsilon_decays)

    if i < len(batch_sizes):
        fbs(batch_sizes[i], it=i+len(learning_rates)+len(epsilon_decays))
        return
    else:
        i -= len(batch_sizes)

    if i < len(target_updates):
        ftu(target_updates[i], it=i+len(learning_rates)+len(epsilon_decays)+len(batch_sizes))
        return
    else:
        i -= len(target_updates)


def f2(i):
    if i >= 4:
        tf.config.set_visible_devices([], 'GPU') # only run 4 on gpu
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    if i < len(batch_sizes):
        fbs2(batch_sizes[i], it=i)
    elif i-len(batch_sizes) < len(replay_memories2):
        frm(replay_memories2[i-len(batch_sizes)], it=i)
    else:
        frm2(replay_memories[i-len(replay_memories2)-len(batch_sizes)], it=i)


version = 2
if __name__ == "__main__":
    # replay_memory_test2 replaces current 32-512 since min replay size was wrong....
    #learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    #epsilon_decays = [0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995]
    #batch_sizes    = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    #target_updates = [5, 10, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000] # number of TERMINAL states before update
    learning_rates = []
    epsilon_decays = []
    batch_sizes    = list(reversed([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    target_updates = []
    replay_memories = [32, 64, 128, 512, 1000, 2500, 5000, 7500, 10_000, 12_500, 15_000, 20_000, 25_000, 50_000]
    replay_memories2 = [32, 64, 128, 512]


    #total_length = len(learning_rates)+len(epsilon_decays)+len(batch_sizes)+len(target_updates)
    total_length = len(replay_memories2)+len(replay_memories)+len(batch_sizes)

    #tf.config.set_visible_devices([], 'GPU')

    #with Pool(24) as p:
    #    p.map(f, list(range(total_length)))
    
    with Pool(18) as p:
        p.map(f2, list(range(total_length)))
    
    