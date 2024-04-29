from models import FFN
from simulator.game15 import Grid
from util.enums import *
import torch.optim as optim
import argparse




class REINFORCE:
    def __init__(self, model, max_episodes):
        self.model = model
        self.max_episodes = max_episodes

    def train_reinforce(model, env, reward, shuffle_cap = None, max_steps = 100, step_frequency = 1000, optimizer="adam", learning_rate=.01, gamma = 1):
        out_actions = 4

        env.reset()
        shuffle = shuffle_cap if episode // step_frequency + 1 > shuffle_cap else episode // step_frequency + 1
        while env.is_solved():
            env.shuffle(shuffle) 

        moves= [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        actions= [0, 1, 2, 3]

        opt = None
        if optimizer is "adam":
            opt = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer is "sgd":
            opt = optim.SGD(model.parameters(), lr=learning_rate)

        for _ in range(max_episodes):
            states = []
            reward_buffer = []
            log_probs = []

            for _ in range(max_steps):
                state = env.get_state()
                states.append(state)
                
                action_dist = torch.softmax(model(state)) # depends on if model output is logits or values
                action = np.random.choice(actions, p = action_dist.numpy())

                log_probs.append(torch.log(action_dist[action]))
        
                r = env.process_move(moves[action])
                reward_buffer.append(r)

                if env.is_solved():
                    break

            reward_buffer = torch.tensor(reward_buffer)
            log_probs = torch.tensor(log_probs)


            # reinforce loss
            discounts = torch.pow(gamma, torch.arange(len(reward_buffer)))
            discounted_rewards = discounts * reward_buffer

            #norm = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards) + 1e-9)

            loss = -torch.sum(log_probs * discounted_rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return model
