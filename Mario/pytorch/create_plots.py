from multiprocessing import Pool
import os
from pathlib import Path
import matplotlib.pyplot as plt

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import numpy as np
from tqdm import tqdm

from agent import Mario
from wrappers import ResizeObservation, SkipFrame

def main(key, i):
    dense_layers = int(key.split('_')[-1])
    data_file_name = f'{key}.txt'
    create_data_file(data_file_name)

    episodes_trained = []
    mean_rewards = []
    mean_steps = []

    for model in files[key]:
        episodes_trained_on_model, model_path = model
        avg_reward, avg_steps = test(f"{key}_{episodes_trained_on_model}", model_path, dense_layers, i)

        episodes_trained.append(episodes_trained_on_model)
        mean_rewards.append(avg_reward)
        mean_steps.append(avg_steps)

        log_data(data=(episodes_trained_on_model, avg_reward, avg_steps), file_name=data_file_name)

    save_plot(f"{key}_reward.jpg", episodes_trained, mean_rewards, "Episodes Trained", "Mean Reward")
    save_plot(f"{key}_steps.jpg", episodes_trained, mean_steps, "Episodes Trained", "Mean Steps")
    
def save_plot(file_name, x, y, xlabel, ylabel):
    plt.plot(x, y)
    plt.grid()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(f"data/{file_name}")
    plt.clf()


def test(name, model_path, dense_layers, i):
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

    checkpoint = Path(model_path)
    mario = Mario(env=env, exploration_rate_min=0.0001, checkpoint=checkpoint, dense_layer=dense_layers)
    mario.exploration_rate = mario.exploration_rate_min

    total_reward = 0
    steps = 0

    for e in tqdm(range(num_episodes), ascii=True, unit='episodes', position=i, desc=name):

        state = np.array(env.reset())

        while True:
            action = mario.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)
            done = done #or info['flag_get']

            total_reward += reward
            steps += 1

            state = next_state

            if done:
                break
        
    avg_reward = total_reward/num_episodes
    avg_steps = steps/num_episodes

    return (avg_reward, avg_steps) 

def log_data(data, file_name):
    # data format: (trained_episodes, mean reward, mean steps) 
    with open(f"data/{file_name}", "a") as f:
        f.write(
            f"{data[0]:8d}{data[1]:15.3f}{data[2]:15.3f}\n"
        )
     
def create_data_file(file_name):
    if not os.path.isdir("data"):
        save_dir = Path("data")
        save_dir.mkdir(parents=True)

    with open(f"data/{file_name}", "w") as f:
        f.write(
            f"{'Episodes':>8}{'MeanReward':>15}{'MeanSteps':>15}\n"
        )

def get_models():
    files = {}
    for dirnames in os.listdir(directory):
        subdir = os.path.join(directory, dirnames)
        for date in os.listdir(subdir):
            subdir = os.path.join(subdir, date)
            for filename in os.listdir(subdir):
                if filename.endswith(".chkpt"):
                    file_path = os.path.join(subdir, filename)
                    episodes_trained_on_model = int(filename.replace(".chkpt","").replace("mario_net_",""))
                    if dirnames not in files: # empty
                        files[dirnames] = [(episodes_trained_on_model, file_path),]
                    else:
                        files[dirnames].append((episodes_trained_on_model, file_path))

        files[dirnames].sort()
        if len(files[dirnames]) != 15:
            print(f'Not 15 checkpoints for {dirnames}, only {len(files[dirnames])}!')
            ans = input(f'Do you want to continue anyway?\n')
            if ans.lower() not in ['y', 'yes']:
                exit()

    return files

def run(i):
    main(key=keys[i], i=i)
                 
     
directory = 'checkpoints'
num_episodes = 10
files = get_models() # format: "folder name" : [(episodes trained, file location), ..]
keys = list(files.keys())

ans = input(f'Are the these correct folders: {keys}?\nLength: {len(keys)}\n')
if ans.lower() not in ['y', 'yes']:
    exit()

with Pool(min(len(keys), 24)) as p:
    p.map(run, list(range(len(keys))))
    


