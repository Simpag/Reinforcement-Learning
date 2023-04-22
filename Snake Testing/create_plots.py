from multiprocessing import Pool
import os
from pathlib import Path
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import numpy as np

def save_plot(location, data):
    # data: ('title', episodes, mean reward, mean steps)
    plt.rcParams["figure.figsize"] = (15,10)
    plt.rcParams.update({'font.size': 22})

    # Reward plots
    for test_data in data:
        name = test_data[0]
        episodes = test_data[1]
        mean_reward = test_data[2]
        plt.plot(episodes, mean_reward, label=name)

    plt.grid()
    #plt.tight_layout()
    plt.xticks([1000*(2*i+1) for i in range(8)])
    plt.xlim((1_000,15_000))
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.ylabel('Mean Reward')
    plt.xlabel('Episodes')
    plt.savefig(f'{location} reward.png', bbox_inches="tight")
    plt.clf()

    # Steps plots
    for test_data in data:
        name = test_data[0]
        episodes = test_data[1]
        mean_steps = test_data[3]
        plt.plot(episodes, mean_steps, label=name)

    plt.grid()
    #plt.tight_layout()
    plt.xticks([1000*(2*i+1) for i in range(8)])
    plt.xlim((1_000,15_000))
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.ylabel('Mean Steps')
    plt.xlabel('Episodes')
    plt.savefig(f'{location} steps.png', bbox_inches="tight")
    plt.clf()

    # Step value
    for test_data in data:
        name = test_data[0]
        episodes = test_data[1]
        mean_reward = np.array(test_data[2])
        mean_steps = np.array(test_data[3])
        y = mean_reward/mean_steps
        plt.plot(episodes, y, label=name)

    plt.grid()
    #plt.tight_layout()
    plt.xticks([1000*(2*i+1) for i in range(8)])
    plt.xlim((1_000,15_000))
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.ylabel('Step Value')
    plt.xlabel('Episodes')
    plt.savefig(f'{location} step value.png', bbox_inches="tight")
    plt.clf()

    # Mean Step value
    for test_data in data:
        name = test_data[0].split()[-1]
        mean_reward = np.array(test_data[2])
        mean_steps = np.array(test_data[3])
        y = np.mean(mean_reward/mean_steps)
        plt.bar(name, y)
    
    plt.xticks(rotation=-45)
    plt.grid()
    #plt.tight_layout()
    plt.ylabel('Step Value')
    plt.xlabel('Episodes')
    plt.savefig(f'{location} mean step value.png', bbox_inches="tight")
    plt.clf()



def get_data(file_collection):
    data = {}
    for test, files in file_collection.items():
        name = test.replace('data/','').replace('_', ' ')
        data[name] = []

        for variable, file_path in files:
            episodes, mean_reward, mean_steps = get_data_from_file(file_path)
            data[name].append((f'{name}: {variable}', episodes, mean_reward, mean_steps))

    return data

def get_data_from_file(file_name):
    # data format: (trained_episodes, mean reward, mean steps) 
    episodes = []
    mean_reward = []
    mean_steps = []
    with open(file_name, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue

            data = list(map(float, line.split()))
            episodes.append(data[0])
            mean_reward.append(data[1])
            mean_steps.append(data[2])

    return episodes, mean_reward, mean_steps
        

def get_files():
    files = {}
    for dirnames in os.listdir(directory):
        subdir = os.path.join(directory, dirnames)
        for filename in os.listdir(subdir):
            if filename.endswith(".txt"):
                file_path = os.path.join(subdir, filename)
                
                variable = filename.split('_')
                variable = float(variable[-1].replace('.txt','')[2:])

                if subdir not in files: # empty
                    files[subdir] = [(variable, file_path)]
                else:
                    files[subdir].append((variable, file_path))

        files[subdir].sort()

    return files



plot_locations = 'data_plots'                
directory = 'data'
files = get_files()
keys = list(files.keys())


ans = input(f'Are the these correct folders: {keys}?\nLength: {len(keys)}\n')
if ans.lower() not in ['y', 'yes']:
    exit()

data = get_data(files) # format: {'test name', ('title', episodes, mean reward, mean steps)}

with alive_bar(len(data)) as bar:
    for test_name, test_data in data.items():
        save_plot(location=f'{plot_locations}/{test_name}', data=test_data)
        bar()
        

