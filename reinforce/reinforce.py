from simulator.game15 import Grid
from util.enums import *
import torch.optim as optim
import argparse
import torch
import numpy as np


class LearnReinforce:
    def __init__(self, model, max_episodes, grid_size):
        self.model = model
        self.max_episodes = max_episodes
        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_reinforce(self, env, opitimizer = "adam", shuffle_cap = None, max_episodes = 10000, max_steps = 100, step_frequency = 1000, optimizer="adam", learning_rate=.01, gamma = 1):

        
        moves= [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        actions= [0, 1, 2, 3]

        opt = None
        if optimizer == "adam":
            opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=learning_rate)

        for episode in range(max_episodes):

            env.reset()
            shuffle = shuffle_cap if episode // step_frequency + 1 > shuffle_cap else episode // step_frequency + 1
            while env.is_solved():
                env.shuffle(shuffle) 

            states = []
            reward_buffer = torch.tensor([], device =self.device, dtype=torch.float32)
            log_probs = torch.tensor([], device =self.device, dtype=torch.float32)
            

            for s in range(max_steps):
                state = env.get_state()
                states.append(state)
                
                outs = self.model(self.preprocess(state).to(self.device, dtype=torch.float32)) # depends on if model output is logits or values
                action_dist = torch.nn.functional.softmax(outs)
                action_dist = torch.squeeze(action_dist, 0)

                action = np.random.choice(actions, p = action_dist.clone().detach().numpy())
                log_prob = torch.log(action_dist[action]).unsqueeze(0).to(self.device, dtype=torch.float32)

                log_probs = torch.cat((log_probs, log_prob))
        
                r = env.process_action(moves[action])
                r = torch.tensor(r).unsqueeze(0).to(self.device, dtype=torch.float32)

                reward_buffer = torch.cat((reward_buffer, r))

                if(s%10 == 0):
                    print(state, action, action_dist)

                if env.is_solved():
                    print("SOLVED")
                    break           
            

            discounted_rewards = torch.empty_like(reward_buffer)
            # reinforce loss
            R = 0
            for i in reversed(range(len(reward_buffer))):
                discounted_rewards[i] = gamma * R + reward_buffer[i]

            
            opt.zero_grad()
            loss = -torch.sum(log_probs * discounted_rewards)
            loss.backward()
            opt.step()

    def test(self, env, max_steps = 100, max_shuffle = 20, set_shuffle = False):
        env.reset()
        total_reward = 0
        steps = 0

        par = np.random.randint(1, max_shuffle) if not set_shuffle else max_shuffle
        while env.is_solved():
            env.shuffle(par)
        state = env.get_state()
        
        while steps < max_steps:
            steps += 1
            with torch.no_grad():
                outs = self.model(self.preprocess(state).to(self.device, dtype=torch.float32)) # depends on if model output is logits or values
                action_dist = torch.nn.functional.softmax(outs)
                action_dist = torch.squeeze(action_dist, 0)

                action = np.random.choice(actions, p = action_dist.clone().detach().numpy())
                
            reward = env.process_action(action)
            state = env.get_state()
            total_reward += reward
            if env.is_solved():
                break

        if steps < max_steps:
            if verbose:
                tmp = "worse" if steps - par > 0 else "better"
                print(f"Completed in {steps} steps, {steps - par if steps - par > 0 else par - steps} steps {tmp} than par of {par}, Total Reward: {total_reward}")
            return steps
        else:
            if verbose:
                print(f"Timed out in {steps} steps")
            return -1

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)
        print(f"Saved model to {file}")

    def run_tests(self, env, num_tests, step_limit = 100, max_shuffle = 20, verbose = True, set_shuffle = False):
        steps = []
        num_timed_out = 0
        for _ in range(num_tests):
            steps.append(self.test(env, step_limit, max_shuffle, verbose, set_shuffle=set_shuffle))
            if steps[-1] == -1:
                steps.pop()
                num_timed_out += 1
        print(f"avg steps = {sum(steps) / len(steps) if len(steps) != 0 else -1} over {num_tests} trials")
        print(f"{num_timed_out} tests timed out")

    def preprocess(self, state):
        # input is gridsize^2 size tuple
        # differs basing on one hot encoding and whether we are using cnns(unsqeeze)
        # currently one hot encoded and made for CNNs
        state_tensor = np.zeros((self.grid_size * self.grid_size, self.grid_size * self.grid_size))
        for i in range(self.grid_size * self.grid_size):
            state_tensor[i, state[i]] = 1
        return torch.tensor(state_tensor, dtype=torch.float).reshape(self.grid_size * self.grid_size, self.grid_size, self.grid_size).unsqueeze(0)


