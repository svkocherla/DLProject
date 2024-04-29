from policies import * 
from simulator.game15 import Grid
from tqdm.auto import tqdm
import torch
from util.utils import loadQNetFromConfig
import argparse


policy_classes = {
    "NNPolicy": NNPolicy,
    "RayPolicy": RayPolicy,
    "QPolicy": QPolicy,
}

parser = argparse.ArgumentParser("Evaluate RL agent on Puzzle")

parser.add_argument("-n", default=4, type=int, help="n x n game board")

parser.add_argument("--checkpoint", type=str, help="path to model checkpoint")
parser.add_argument("--policy-class", default="NNPolicy", type=str, choices=list(policy_classes.keys()), help="wrapper class")

parser.add_argument("--n-games", default=100, type=int, help="number of games to evaluate")
parser.add_argument("--n-shuffles", default=10, type=int, help="number of moves to shuffle")
parser.add_argument("--max-moves", default=10**5, type=int, help="max length of a trajectory")

parser.add_argument("-v", "--verbose", default=False, action="store_true", help="show progress bar")

def play(grid: Grid, policy: Policy, max_moves=10000):
    '''
        Play game until the model makes an invalid move, solves the game, or times out
        returns list of moves played by the model
    '''
    moves = []
    for _ in range(max_moves):
        move = policy.get_move(grid)
        if grid.process_move(move):
            moves.append(move)
        else:
            break

        if grid.is_solved():
            break
    return moves

def evaluate(env: Grid, policy, n_games=10, max_moves=10000, n_shuffles=10, verbose=False):
    games = range(n_games)
    if verbose: 
        games = tqdm(games, desc="Playing Games")

    n_solved = 0
    for _ in games:
        env.reset()
        env.shuffle(n_shuffles)
        play(env, policy, max_moves)
        if env.is_solved():
            n_solved+=1
    success = float(n_solved * 100)/n_games
    return success


def main():
    args = parser.parse_args()
    
    # initialize simulator
    # grid = Grid(args.n)

    # load model
    # model = torch.load(args.checkpoint) #"checkpoints/ray_test/model.pt"
    env, dqn, train_test = loadQNetFromConfig(f'q_learning/model_configs/Qnet4x4.json')
    policy_class = policy_classes[args.policy_class]
    # policy = policy_class(model)
    policy = policy_class(dqn)

    success = evaluate(env, policy, n_games=args.n_games, n_shuffles=args.n_shuffles, max_moves=args.max_moves, verbose=args.verbose)
    print(f"{success:.2f}% success rate")

if __name__ == "__main__":
    main()