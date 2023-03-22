import getopt, sys


def start_test(file_name, env_name, sleep_time):
    import random
    import keyboard
    import time
    from sys import platform
    import numpy as np
    import tensorflow as tf
    from keras.models import load_model
    from alive_progress import alive_bar

    # Mario stuff
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    # Env wrappers
    from EnvWrappers.wrappers import SkipFrame
    from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

    #tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)
    
    # Create the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT) 

    # apply wrappers to the env (increase training speed)
    env = SkipFrame(env=env, skip=4)                    # Skip every 4 frames
    env = GrayScaleObservation(env=env, keep_dim=True)  # Turn image into grayscale
    env = ResizeObservation(env=env, shape=84)          # Resize 240x256 image to 84x84
    env = FrameStack(env=env, num_stack=4)              # Stack 4 frames

    # Load the model
    model = load_model(file_name)

    # Define the testing function
    def test(episodes):
        total_reward = 0
        episodes_skipped = 0
        with alive_bar(episodes) as bar:
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
                    env.render('human')
                    if sleep_time is not None:
                        time.sleep(sleep_time)

                    if platform == "win32":
                        if keyboard.is_pressed("q"):
                            print(f"Skipping episode {episode}")
                            done = True
                            episodes_skipped += 1
                            episode_reward = 0
                            time.sleep(0.5) # So we dont skip mulitple episodes

                total_reward += episode_reward
                bar()

        env.close()
        return total_reward / (episodes - episodes_skipped)

    # Test the trained model
    average_reward = test(2)
    print("Average reward:", average_reward)


if __name__ == "__main__":
    argumentList = sys.argv[1:]
    
    # Options
    options = "hf:s:"
    
    # Long options
    long_options = ["Help", "File=", "Fps="]

    env_name = None
    file_name = None
    sleep_time = None

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--Help"):
                print("python start_testing.py -f FILE_NAME -e ENV_NAME")
                print("Example: python start_testing.py -f models/Snake_16x16_____9.00max____0.18avg___-1.00min__1675760303.model/")
            elif currentArgument in ("-f", "--File"):
                file_name = currentValue
            elif currentArgument in ("-e", "--Env"):
                env_name = currentValue
            elif currentArgument in ("-s", "--Fps"):
                sleep_time = 1/float(currentValue)

        if file_name is not None:
            start_test(file_name, env_name, sleep_time)
                
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))