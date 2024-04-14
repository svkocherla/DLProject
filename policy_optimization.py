from simulator.game15 import Grid
from util.enums import *
import torch
import torch.optim as optim
import numpy as np

# states, actions, rewards, transitions, initial state distribution
def reinforce(game, model, reward, optimizer, episodes=10):
    # actions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    actions = [0, 1, 2, 3]

    for _ in range(episodes):
        states = []
        action_buffer = []
        reward_buffer = []

        while not game.is_solved():
            state = game.get_state()
            action_dist = torch.softmax(model(state)) # depends on if model output is logits or values
            action = np.random.choice(actions, p = action_dist.numpy())
            action_buffer.append(action)
        
            r = reward(state, action)
            reward_buffer.append(reward)

            game.process_move(action)

        reward_buffer = torch.tensor(reward_buffer)

    # NOTE: David - I have no idea what I'm doing. 
    # Haven't read deeply into REINFORCE yet. 
    # Putting total_loss variable here just to test training loop.
    total_loss = 0
    for step in range(len(states)):
        future_rewards = torch.sum(reward_buffer[step:])

        state = states[step]
        action = action_buffer[step]

        log_prob = torch.log(model(state))[action]
        loss = -log_prob * future_rewards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss

    return total_loss


def ppo(model, reward, optimizer, learning_rate=.01, gamma = .9, epsilon =.2, clip = .1):
    out_actions = 4

    game = Grid()

    # actions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    actions = [0, 1, 2, 3]

    for _ in range(episodes):
        states = []
        action_buffer = []
        reward_buffer = []
        value_buffer = []

        while not game.is_solved():
            state = game.get_state()
            logits = model(state)
            action_dist = torch.softmax(logits) # depends on if model output is logits or values
            action = np.random.choice(actions, p = action_dist.numpy())
            action_buffer.append(action)
        
            r = reward(state, action)
            reward_buffer.append(reward)

            value = torch.max(logits)

            value_buffer.append(value)

            game.process_move(action)

        reward_buffer = torch.tensor(reward_buffer)
        value_buffer = torch.tensor(value_buffer)
        returns = calculate_ppo_returns(returns, gamma)

        adv = returns - value_buffer

        adv = (adv - torch.mean(adv))/ torch.std(adv)

        update_ppo(model, optimizer, states, action_buffer, returns, adv, epsilon, clip)


    for step in range(len(states)):
        future_rewards = torch.sum(reward_buffer[step:])

        state = states[step]
        action = action_buffer[step]

        log_prob = torch.log(model(state))[action]
        loss = -log_prob * future_rewards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def calculate_ppo_returns(reward_buffer, gamma):
    returns = []
    R = 0
    for r in range(len(reward_buffer)):
        reward = reward_buffer[len(reward_buffer)-r-1]
        R = reward + gamma * R
        returns.append(R)

    returns = reversed(returns)
    return torch.tensor(returns)

def update_ppo(model, optimizer, states, actions_buffer, returns, advantages, epsilon, clip_threshold):
    model.eval()
    
    logits = torch.tensor()
    
    for observation in states:
        output = model(state)
        logits = torch.vstack(logits, output)

    probs = torch.softmax(logits, dim=-1)
    action_probs = probs.gather(1, actions_buffer.unsqueeze(1))
        
    old_logits = logits.detach()
    old_probs = torch.softmax(old_logits, dim=-1).detach()
    old_action_probs = old_probs.gather(1, action_buffer.unsqueeze(1))

    ratio = action_probs / old_action_probs
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))



    thresh = old_action_probs + torch.clamp(probs - old_probs, -clip_threshold, clip_threshold)
    value_loss = torch.mean(torch.max((returns - clipped_values)**2, (returns - old_action_probs)**2))

    total_loss = actor_loss + value_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()