from multiprocessing import Pool
import os
from pathlib import Path
import matplotlib.pyplot as plt

def main(key, i):
    import gym
    import gym_snake
    from keras.models import load_model
    from tqdm import tqdm

    import tensorflow as tf
    import numpy as np

    def get_action(model, state):
        state_tensor = tf.convert_to_tensor(np.array(state)/255)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        return action

    def test(name, model_path, i):
        tf.config.set_visible_devices([], 'GPU')
        
        env = gym.make("Snake-16x16-8a-v0") 

        env.reset()

        model = load_model(model_path)

        total_reward = 0
        steps = 0

        for e in tqdm(range(num_episodes), ascii=True, unit='episodes', position=i, desc=name):

            state = np.array(env.reset())

            while True:
                action = get_action(model, state)

                next_state, reward, done, info = env.step(action)
                next_state = np.array(next_state)

                total_reward += reward
                steps += 1

                state = next_state

                if done:
                    break
            
        avg_reward = total_reward/num_episodes
        avg_steps = steps/num_episodes

        env.close()

        return (avg_reward, avg_steps) 



    data_file_name = f'{key}.txt'
    create_data_file(data_file_name)

    episodes_trained = []
    mean_rewards = []
    mean_steps = []

    for model in files[key]:
        episodes_trained_on_model, model_path = model
        avg_reward, avg_steps = test(f"{key}_{episodes_trained_on_model}", model_path, i)

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
        for model_test in os.listdir(subdir):
            model_subdir = os.path.join(subdir, model_test)
            name = dirnames+'_'+model_test
            for filename in os.listdir(model_subdir):
                if filename.endswith(".model"):
                    file_path = os.path.join(model_subdir, filename)
                    episodes_index_start = filename.find('episode_')+len('episode_')
                    episodes_index_end = episodes_index_start + filename[episodes_index_start:].find('_')
                    episodes_trained_on_model = int(filename[episodes_index_start:episodes_index_end])
                    
                    if name not in files: # empty
                        files[name] = [(episodes_trained_on_model, file_path),]
                    else:
                        files[name].append((episodes_trained_on_model, file_path))
            
            files[name].sort()

    return files

def run(i):
    main(key=keys[i], i=i)
                 
     
directory = 'models/HParams/15k episodes'
num_episodes = 10
files = get_models() # format: "folder name" : [(episodes trained, file location), ..]
keys = list(files.keys())

ans = input(f'Are the these correct folders: {keys}?\nLength: {len(keys)}\n')
if ans.lower() not in ['y', 'yes']:
    exit()

with Pool(min(len(keys), 24)) as p:
    p.map(run, list(range(len(keys))))



