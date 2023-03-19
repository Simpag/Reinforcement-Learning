import random
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from DDQN.CERDDQNAgent import CERDDQNAgent
from DDQN.DDQNAgent import DDQNAgent

# Mario stuff
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Env wrappers
from EnvWrappers.wrappers import SkipFrame
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

# To enable multiple processes or something idk, just run it before the script
# export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

def main(profiler = None):
    # Model settings
    MODEL_NAME = "Testing_improvements"
    MODEL_TO_LOAD = None                # Load model from file, (None = wont load)
    TARGET_MODEL_UPDATE_CYCLE = 5       # Number of terminal states before updating target model
    REPLAY_MEMORY_SIZE = 25_000         # How big the batch size should be
    MIN_REPLAY_MEMORY_SIZE = 1_000      # Number of steps recorded before training starts

    # Training settings
    STARTING_EPISODE = 1                # Which episode to start from (should be 1 unless continued training on a model)
    EPISODES = 100#20_000                   # Total training episodes
    MINIBATCH_SIZE = 32                 # How many steps to use for training

    #  Stats settings
    MIN_REWARD = 10**1000               # Save model that reaches min avg reward
    AGGREGATE_STATS_EVERY = 50          # When to record data to plot how it performs
    SHOW_PREVIEW = False                # Show preview of agent playing

    # DQ-settings
    DISCOUNT = 0.99                     # gamma (discount factor)
    LEARNING_RATE = 0.001

    # Exploration settings
    epsilon = 1                         # Not a constant, going to be decayed
    EPSILON_DECAY = 0.9999 #0.99975 #0.95
    MIN_EPSILON = 0.001

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)

    # For stats
    ep_rewards = [-1,] 

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT) 

    # apply wrappers to the env (increase training speed)
    env = SkipFrame(env=env, skip=4)                    # Skip every 4 frames
    env = GrayScaleObservation(env=env, keep_dim=True)  # Turn image into grayscale
    env = ResizeObservation(env=env, shape=84)          # Resize 240x256 image to 84x84
    env = FrameStack(env=env, num_stack=4)              # Stack 4 frames

    # ENV: gym.Env, DISCOUNT: float, LEARNING_RATE: int, TARGET_MODEL_UPDATE_CYCLE: int, REPLAY_MEMORY_SIZE: int, MINIBATCH_SIZE: int, MIN_REPLAY_MEMORY_SIZE: int, MODEL_NAME: str, MODEL_TO_LOAD = None, LOG_DIR = None
    agent = CERDDQNAgent(env, DISCOUNT, LEARNING_RATE, TARGET_MODEL_UPDATE_CYCLE, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, MODEL_NAME, MODEL_TO_LOAD)

    for episode in tqdm(range(STARTING_EPISODE, EPISODES + 1), ascii=True, unit='episodes'):
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
            if np.random.random() > epsilon:
                # Get action from DQN
                action = agent.get_action(current_state)
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, info = env.step(action)

            if info['flag_get']:
                done = True

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

            if profiler is not None:
                if len(agent.replay_memory) >= agent.MIN_REPLAY_MEMORY_SIZE:
                    profiler.enable()


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

if __name__ == "__main__":
    profile = False
    
    if profile: # if you want to save a profiling log
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        #profiler.enable()
        main(profiler=profiler)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats('profiles/p.dat')
        # snakeviz profiles/p.dat
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        main()