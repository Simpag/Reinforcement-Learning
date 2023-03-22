from multiprocessing import Pool
import numpy as np
import gym
import gym_snake
from keras.models import load_model
from alive_progress import alive_bar
import random
import tensorflow as tf
import time
from keras import backend as K
import gc
import matplotlib.pyplot as plt

def test(model, env, episodes):
    total_reward = 0
    with alive_bar(episodes, title="Model") as bar:
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                state_tensor = tf.convert_to_tensor(np.array(state)/255)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()
                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_reward += reward
                #env.render()
            bar()

            total_reward += episode_reward

    env.close()
    return total_reward / (episodes)

def start_test(file_name, x_val, env_name, episodes):
    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)
    
    # Create the environment
    env = gym.make(env_name) 

    # Load the model
    model = load_model(file_name)    

    # Test the trained model
    average_reward = test(model, env, episodes)

    del model
    gc.collect()
    K.clear_session()

    #print("Average reward:", average_reward)
    return (x_val, average_reward)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    env_name = "Snake-16x16-8a-v0"
    ### Epsilon decay
    if False:
        name = "Epsilon Decay"
        models = [
            "models/epsilon_decay_test/16x16_8a_ed0.8/16x16_8a_lr0.001_ed0.8_episode_5000_0.001epsilon_1677687457.model",
            "models/epsilon_decay_test/16x16_8a_ed0.9/16x16_8a_lr0.001_ed0.9_episode_5000_0.001epsilon_1677691161.model",
            "models/epsilon_decay_test/16x16_8a_ed0.95/16x16_8a_lr0.001_ed0.95_episode_5000_0.001epsilon_1677695034.model",
            "models/epsilon_decay_test/16x16_8a_ed0.97/16x16_8a_lr0.001_ed0.97_episode_5000_0.001epsilon_1677699424.model",
            "models/epsilon_decay_test/16x16_8a_ed0.98/16x16_8a_lr0.001_ed0.98_episode_5000_0.001epsilon_1677703402.model",
            "models/epsilon_decay_test/16x16_8a_ed0.99/16x16_8a_lr0.001_ed0.99_episode_5000_0.001epsilon_1677707196.model",
            "models/epsilon_decay_test/16x16_8a_ed0.995/16x16_8a_lr0.001_ed0.995_episode_5000_0.001epsilon_1677711415.model",
            "models/epsilon_decay_test/16x16_8a_ed0.999/16x16_8a_lr0.001_ed0.999_episode_5000_0.006727839799665273epsilon_1677716309.model",
            "models/epsilon_decay_test/16x16_8a_ed0.9995/16x16_8a_lr0.001_ed0.9995_episode_5000_0.082074731797801epsilon_1677721234.model",
            "models/epsilon_decay_test/16x16_8a_ed0.9999/16x16_8a_bs32_lr0.001_ed0.9999_tu10_episode_5000_0.6065761532401006epsilon_1677727963.model",
                ] 
        model_x = [
            0.8, 
            0.9, 
            0.95, 
            0.97, 
            0.98, 
            0.99, 
            0.995, 
            0.999, 
            0.9995, 
            0.9999, 
        ]

    ### Learning Rate
    if False:
        name = "Learning Rate"
        models = [
            "models/learning_rate_test/16x16_8a_lr0.1/16x16_8a_lr0.1_episode_5000_0.6065761532401006epsilon_1677682328.model",
            "models/learning_rate_test/16x16_8a_lr0.01/16x16_8a_lr0.01_episode_5000_0.6065761532401006epsilon_1677667966.model",
            "models/learning_rate_test/16x16_8a_lr0.001/16x16_8a_lr0.001_episode_5000_0.6065761532401006epsilon_1677653612.model",
            "models/learning_rate_test/16x16_8a_lr0.0001/16x16_8a_lr0.0001_episode_5000_0.6065761532401006epsilon_1677639861.model",
            "models/learning_rate_test/16x16_8a_lr0.02/16x16_8a_lr0.021544346900318822_episode_5000_0.6065761532401006epsilon_1677672686.model",
            "models/learning_rate_test/16x16_8a_lr0.002/16x16_8a_lr0.002154434690031882_episode_5000_0.6065761532401006epsilon_1677658360.model",
            "models/learning_rate_test/16x16_8a_lr0.0002/16x16_8a_lr0.00021544346900318845_episode_5000_0.6065761532401006epsilon_1677644315.model",
            "models/learning_rate_test/16x16_8a_lr0.04/16x16_8a_lr0.046415888336127774_episode_5000_0.6065761532401006epsilon_1677677609.model",
            "models/learning_rate_test/16x16_8a_lr0.004/16x16_8a_lr0.004641588833612777_episode_5000_0.6065761532401006epsilon_1677663125.model",
            "models/learning_rate_test/16x16_8a_lr0.0004/16x16_8a_lr0.00046415888336127773_episode_5000_0.6065761532401006epsilon_1677648979.model",
        ]
        model_x = [
            0.1,
            0.01,
            0.001,
            0.0001,
            0.02,
            0.002,
            0.0002,
            0.046,
            0.0046,
            0.000046,
        ]

    ### Batch Sizes
    if False:
        name = "Batch Size"
        models = [
            "models/batch_size_test/16x16_8a_bs1/16x16_8a_bs1_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677763412.model",
            "models/batch_size_test/16x16_8a_bs4/16x16_8a_bs4_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677762736.model",
            "models/batch_size_test/16x16_8a_bs8/16x16_8a_bs8_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677763160.model",
            "models/batch_size_test/16x16_8a_bs16/16x16_8a_bs16_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677763470.model",
            "models/batch_size_test/16x16_8a_bs32/16x16_8a_bs32_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677763413.model",
            "models/batch_size_test/16x16_8a_bs64/16x16_8a_bs64_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677763703.model",
            "models/batch_size_test/16x16_8a_bs128/16x16_8a_bs128_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677764159.model",
            "models/batch_size_test/16x16_8a_bs256/16x16_8a_bs256_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677764450.model",
            "models/batch_size_test/16x16_8a_bs512/16x16_8a_bs512_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677765194.model",
        ]
        model_x = [
            1, 
            4, 
            8, 
            16, 
            32, 
            64, 
            128, 
            256, 
            512
        ]

    ### Target Update
    if True:
        name = "Target Update"
        models = [
            "models/target_update_test/16x16_8a_tu5/16x16_8a_bs32_lr0.001_ed0.9995_tu5_episode_5000_0.082074731797801epsilon_1677770722.model",
            "models/target_update_test/16x16_8a_tu10/16x16_8a_bs32_lr0.001_ed0.9995_tu10_episode_5000_0.082074731797801epsilon_1677771106.model",
            "models/target_update_test/16x16_8a_tu25/16x16_8a_bs32_lr0.001_ed0.9995_tu25_episode_5000_0.082074731797801epsilon_1677771105.model",
            "models/target_update_test/16x16_8a_tu50/16x16_8a_bs32_lr0.001_ed0.9995_tu50_episode_5000_0.082074731797801epsilon_1677771454.model",
            "models/target_update_test/16x16_8a_tu75/16x16_8a_bs32_lr0.001_ed0.9995_tu75_episode_5000_0.082074731797801epsilon_1677771800.model",
            "models/target_update_test/16x16_8a_tu100/16x16_8a_bs32_lr0.001_ed0.9995_tu100_episode_5000_0.082074731797801epsilon_1677772110.model",
            "models/target_update_test/16x16_8a_tu250/16x16_8a_bs32_lr0.001_ed0.9995_tu250_episode_5000_0.082074731797801epsilon_1677772490.model",
            "models/target_update_test/16x16_8a_tu500/16x16_8a_bs32_lr0.001_ed0.9995_tu500_episode_5000_0.082074731797801epsilon_1677772932.model",
            "models/target_update_test/16x16_8a_tu1000/16x16_8a_bs32_lr0.001_ed0.9995_tu1000_episode_5000_0.082074731797801epsilon_1677773548.model",
        ]
        model_x = [
            5, 
            10, 
            25, 
            50, 
            75, 
            100, 
            250, 
            500, 
            1000
        ]

    episodes = 100

    results = list()

    def helper(i):
        return start_test(models[i], model_x[i], env_name, episodes)

    with Pool(18) as p:
        results = p.map(helper, list(range(len(models))))

    print(results)
    plt.figure()
    plt.title(f'{name}')
    plt.xlabel(f'{name} Value')
    plt.ylabel('Average Reward')
    X = []
    Y = []
    for i in range(len(models)):
        x,y = results[i]
        X.append(x)
        Y.append(y)
    plt.grid()
    plt.plot(X, Y, '*')
    #plt.legend()
    plt.show()

    