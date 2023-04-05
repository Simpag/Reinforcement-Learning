import random, datetime, time
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import numpy as np
from tqdm import tqdm

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

env = JoypadSpace(
        env,
        [
        ['NOOP'],
        ['right'],
        ['right', 'A']
        ]
    )

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

checkpoint = Path('checkpoints/CER_512/2023-04-03T20-38-04/mario_net_15000.chkpt')
mario = Mario(env=env, exploration_rate_min=0.001, checkpoint=checkpoint, dense_layer=512)
mario.exploration_rate = mario.exploration_rate_min

episodes = 5
total_reward = 0

for e in tqdm(range(episodes), ascii=True, unit='episodes'):

    state = np.array(env.reset())

    while True:

        env.render()

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state)
        done = done #or info['flag_get']

        total_reward += reward

        mario.cache(state, next_state, action, reward, done)

        #logger.log_step(reward, None, None)

        state = next_state

        if done:
            break

        #time.sleep(1/30)

    """logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )"""
    
print(f'Average reward: {total_reward/episodes}')
