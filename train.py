import torch.optim as optim
import argparse
import wandb
from torch.utils.tensorboard import SummaryWriter

from simulator.game15 import Grid
from util.enums import *
from architectures import FFN
import models
import reward
import policy_optimization
from evaluation import evaluate
from policies import *

parser = argparse.ArgumentParser("Train RL agent on Game15")
parser.add_argument("-n", default=3, type=int, help="n x n game board")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--episodes", default=10, type=int, help="# of episodes per epoch")
parser.add_argument("--max-moves", default=10**5, type=int, help="max length of a trajectory")
parser.add_argument("--shuffles", default=10, type=int, help="number of moves to shuffle")

parser.add_argument("--model", default="ffn", type=str, help="model architecture as defined in models.py")
parser.add_argument("--reward-fn", default="naive", type=str, help="reward function as defined in reward.py")
parser.add_argument("--algorithm", default="reinforce", type=str, help="RL algorithm as defined in policy_optimization.py")
parser.add_argument("--eval-batch-size", default=10, type=int, help="# of games to evaluate per epoch")
parser.add_argument("--eval-every", default=1, type=int, help="# of epochs between each evaluation")

parser.add_argument("--run-name", default=10**5, type=str, help="")

def main(args):
    model = models.__dict__[args.model](args.n)

    # TODO: if chkpt, load from checkpoint

    ## training_loop
    game = Grid(args.n)
    model = FFN(args.n, 100) # not sure what game state representation we should use
    optimizer = optim.Adam(model.parameters())
    train = policy_optimization.__dict__[args.algorithm]
    reward_fn = reward.__dict__[args.reward_fn]

    policy = NNPolicy(model)

    for epoch in range(args.epochs):
        # train
        loss = train(game, model, reward_fn, optimizer, args.episodes)
        # if epoch % args.eval_every == 0:
        # # evaluate
        #     success = evaluate(game, policy, 
        #                    evaluations=args.eval_batch_size, 
        #                    max_moves=args.max_moves, 
        #                    n_shuffles=args.shuffles, 
        #                    verbose=False)
        #     # wandb.log({"success": success})
        #     # print()
        print(loss)
        
        # wandb.log({"loss": loss})

    # TODO: save model to checkpoint

if __name__ == "__main__":
    args = parser.parse_args()

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="7643-project",
    #     # track hyperparameters and run metadata
    #     config={
    #     **args
    #     }
    # )

    # writer = SummaryWriter(<path to log>)
    main(args)
    # wandb.finish()