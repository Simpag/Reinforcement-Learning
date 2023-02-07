import os
import numpy as np
import gym
import gym_snake
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras import backend as K
import gc
from datetime import datetime
from alive_progress import alive_bar

#import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Create the environment
env = gym.make("Snake-16x16-big-apple-reward-v0") 
#env = gym.make("Snake-16x16-v0") 

# Get the number of actions and states
num_actions = env.action_space.n
state_size = env.observation_space.shape

# Save the model
def save(model, filename):
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')
    save_model(model, 'models/' + filename)

# Load the model
def load(filename):
    return load_model(filename)

# Define the model
if True:
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=state_size))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))

    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
else:
    model = load('models/snake_model_6000.h5')

# Define the Q-learning algorithm
def q_learning(episodes, iteration):
    epsilon = 0.9
    with alive_bar(episodes, title=f'Run: {iteration}') as bar:
        for episode in range(episodes):
            stepCount = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])
            done = False
            while not done:
                if epsilon > np.random.random(1):
                    action = np.random.randint(0,num_actions)
                else:
                    # Choose an action with the highest predicted Q-value
                    action = np.argmax(model.predict(state, verbose=0))
                
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
                target = reward + 0.95 * np.amax(model.predict(next_state, verbose=0))
                target_vec = model.predict(state, verbose=0)[0]
                target_vec[action] = target
                model.fit(state, target_vec.reshape(-1, num_actions), epochs=1, verbose=0)
                state = next_state

                stepCount += 1

                if stepCount > 100:
                    env.render('human')
            epsilon *= 0.95
            if episode % 100 == 0: 
                gc.collect()        # without this I get mad memory leak
                K.clear_session()   # without this I get mad memory leak
            bar()
        env.close()


for i in range(1,20):
    # Train the model
    q_learning(1000, i)

    # Save the trained model
    #save(model, f"snake_model_{str(datetime.now().replace(microsecond=0)).replace(':', '-').replace(' ', '_')}.h5")
    save(model, f"snake_model_{i*1000}.h5")


# export PATH=/usr/local/cuda/bin:$PATH enable cuda or sm shit
# export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"
