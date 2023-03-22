import getopt, sys


def start_test(file_name, env_name, sleep_time):
    import numpy as np
    import gym
    import gym_snake
    from keras.models import load_model
    from alive_progress import alive_bar
    import random
    import tensorflow as tf
    import keyboard
    import time
    from sys import platform

    tf.config.set_visible_devices([], 'GPU')

    # For more repetitive results
    ENV_SEED = 1
    random.seed(ENV_SEED)
    np.random.seed(ENV_SEED)
    tf.random.set_seed(ENV_SEED)
    
    # Create the environment
    env = gym.make(env_name) 

    # Load the model
    model = load_model(file_name)

    # Define the testing function
    def test(model, episodes):
        total_reward = 0
        episodes_skipped = 0
        with alive_bar(episodes) as bar:
            for episode in range(episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    #action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)/255, verbose=0)[0])
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
    average_reward = test(model, 100)
    print("Average reward:", average_reward)


if __name__ == "__main__":
    argumentList = sys.argv[1:]
    
    # Options
    options = "hf:e:s:"
    
    # Long options
    long_options = ["Help", "File=", "Env=", "Fps="]

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
                print("Example: python start_testing.py -f models/Snake_16x16_____9.00max____0.18avg___-1.00min__1675760303.model/ -e Snake-16x16-v0")
            elif currentArgument in ("-f", "--File"):
                file_name = currentValue
            elif currentArgument in ("-e", "--Env"):
                env_name = currentValue
            elif currentArgument in ("-s", "--Fps"):
                sleep_time = 1/float(currentValue)

        if file_name is not None:
            if env_name is None: 
                print("Wrong input")
                print("python start_testing.py -f FILE_NAME -e ENV_NAME")
                print("Example: python start_testing.py -f models/Snake_16x16_____9.00max____0.18avg___-1.00min__1675760303.model/ -e Snake-16x16-v0")
            else: 
                start_test(file_name, env_name, sleep_time)
                
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))