import numpy as np
import gym
import gym_snake
from keras.models import load_model
from alive_progress import alive_bar


# Create the environment
env = gym.make("Snake-16x16-v0") 

# Get the number of actions and states
num_actions = env.action_space.n
state_size = env.observation_space.shape

# Load the model
model = load_model("models/snake_model_2000.h5")

# Define the testing function
def test(model, episodes):
    total_reward = 0
    with alive_bar(episodes) as bar:
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])
            done = False
            episode_reward = 0
            while not done:
                action = np.argmax(model.predict(state, verbose=0))
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
                state = next_state
                episode_reward += reward
                env.render('human')
            total_reward += episode_reward
            bar()

    env.close()
    return total_reward / episodes

# Test the trained model
average_reward = test(model, 100)
print("Average reward:", average_reward)
