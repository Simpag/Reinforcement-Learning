from multiprocessing import Pool
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import SkipFrame, ResizeObservation

def main(i, name, use_cer):
    # Model settings
    TARGET_MODEL_UPDATE_CYCLE = 1000    # Number of terminal states before updating target model
    REPLAY_MEMORY_SIZE = 50_000         # How big the batch size should be
    DENSE_LAYER_SIZE = 512

    # Training settings
    STARTING_EPISODE = 1                # Which episode to start from (should be 1 unless continued training on a model)
    EPISODES = 40_000                   # Total training episodes
    SAVE_AGENT_EVERY = 1000             # How many episodes before saving agent
    MINIBATCH_SIZE = 32                 # How many steps to use for training
    BURN_IN = 1e5                       # min. experiences before training
    USE_GPU = True
    USE_CER = use_cer

    #  Stats settings
    AGGREGATE_STATS_EVERY = 1           # When to record data to plot how it performs
    PLOT_STATS_EVERY = 1000             # When to plot the stats
    SHOW_PREVIEW = False                # Show preview of agent playing

    # DQ-settings
    DISCOUNT = 0.95                     # gamma (discount factor)
    LEARNING_RATE = 0.00025

    # Exploration settings
    EPSILON = 1                         # Not a constant, going to be decayed
    EPSILON_DECAY = 0.9999995 #0.99975 #0.95
    MIN_EPSILON = 0.01


    ###############
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)

    # Initialize Super Mario environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(
        env,
        [
        ['NOOP'],
        ['right'],
        ['right', 'A']
        ]
    )

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    save_dir = Path('checkpoints') / name / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
    mario = Mario(env=env, discount_factor=DISCOUNT, learning_rate=LEARNING_RATE, target_model_update=TARGET_MODEL_UPDATE_CYCLE, replay_memory_size=REPLAY_MEMORY_SIZE, minibatch_size=MINIBATCH_SIZE, exploration_rate=EPSILON, exploration_rate_decay=EPSILON_DECAY, exploration_rate_min=MIN_EPSILON, use_gpu=USE_GPU, cer_agent=USE_CER, burn_in=BURN_IN, dense_layer=DENSE_LAYER_SIZE, save_dir=save_dir, checkpoint=checkpoint)

    logger = MetricLogger(save_dir)


    ### for Loop that train the model num_episodes times by playing the game
    for e in tqdm(range(STARTING_EPISODE, EPISODES + 1), ascii=True, unit='episodes', position=i, desc=name):
        state = np.array(env.reset())

        # Play the game!
        while True:

            # 3. Show environment (the visual) [WIP]
            if SHOW_PREVIEW:
                env.render('human')

            # 4. Run agent on the state
            action = mario.act(state)

            # 5. Agent performs action
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)

            # 6. Remember
            mario.cache(state, next_state, action, reward, done)

            # 7. Learn
            q, loss = mario.learn()

            # 8. Logging
            logger.log_step(reward, loss, q)

            # 9. Update state
            state = next_state

            # 10. Check if end of game
            if done or info['flag_get']:
                break

        logger.log_episode()

        if e % AGGREGATE_STATS_EVERY == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step,
                aggregate_stats_every=AGGREGATE_STATS_EVERY,
                plot_stats=not e%PLOT_STATS_EVERY,
                print_to_console=False
            )

        # Save the agent
        if not e%SAVE_AGENT_EVERY:
            mario.save(e)


def f(i):
    if i < 1:
        main(i=i, name="CER", use_cer=True)
    else:
        main(i=i, name="Normal", use_cer=False)


if __name__ == "__main__":

    total_length = 2

    with Pool(min(24, total_length)) as p:
        p.map(f, list(range(total_length)))
