import torch
import random, numpy as np
from pathlib import Path
import gym

from neural import MarioNet
from collections import deque

class Mario:
    def __init__(self, env: gym.Env, discount_factor: float=0.9, learning_rate: float=0.00001, target_model_update: int=500, replay_memory_size: int=2.5e4, minibatch_size:int=32, exploration_rate=1, exploration_rate_decay=0.9999999, exploration_rate_min=0.01, use_gpu: bool=False, use_amp: bool= False, cer_agent: bool=False, burn_in: int=1e5, dense_layer: int=512, save_dir=None, checkpoint=None):
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        self.memory = deque(maxlen=int(replay_memory_size))
        self.batch_size = minibatch_size

        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.gamma = discount_factor

        self.curr_step = 0
        self.burnin = burn_in  # min. experiences before training
        self.learn_every = 1   # no. of experiences between updates to Q_online
        self.sync_every = target_model_update #1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e5   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        #self.use_cuda = torch.cuda.is_available()
        self.use_cuda = use_gpu
        self.use_amp = use_amp

        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.CER = cer_agent # if we should use CER for replay memory

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim, dense_layer).float()
        if self.use_cuda and torch.cuda.is_available():
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        if self.CER:
            batch[0] = self.memory[-1] # Take the latest experience into the batch
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            next_state_Q = self.net(next_state, model='online')
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss = self.loss_fn(td_estimate, td_target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()
        """loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        #loss.backward()
        self.optimizer.step()
        return loss.item()"""


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self, end_of_episode):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        #if self.curr_step % self.save_every == 0:
        #    self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        # decrease exploration_rate
        #if end_of_episode:
        #    self.exploration_rate *= self.exploration_rate_decay
        #    self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return (td_est.mean().item(), loss)


    def save(self, episode):
        #save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        save_path = self.save_dir / f"mario_net_{episode}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        #print(f"MarioNet saved to {save_path} at step {self.curr_step}")
        print(f"MarioNet saved to {save_path} at episode {episode}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
