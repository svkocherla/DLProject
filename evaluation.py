# from models import FFN
from policies import NNPolicy, Policy
from simulator.game15 import Grid
from tqdm.auto import tqdm

def play(grid: Grid, policy: Policy, max_moves=10000):
    '''
        Play game until the model makes an invalid move, solves the game, or times out
        returns list of moves played by the model
    '''
    moves = []
    for _ in range(max_moves):
        move = policy.get_move(grid)
        if grid.process_move(move) == "VALID":
            moves.append(move)
        else:
            break

        if grid.is_solved():
            break
    return moves

def evaluate(grid: Grid, policy, n_games=10, max_moves=10000, n_shuffles=10, verbose=False):
    games = range(n_games)
    if verbose: 
        games = tqdm(games, desc="Playing Games")

    n_solved = 0
    for game in games:
        grid.reset()
        grid.shuffle_n(n_shuffles)
        play(grid, policy)
        if grid.is_solved():
            n_solved+=1
    success = float(n_solved * 100)/n_games
    return success

# def main():
#     model = FFN()
#     policy = NNPolicy(model)

#     n_solved = 0

#     # print(f"{float(n_solved * 100)/n_games:.2f}% success rate: {n_solved}/{n_games} games solved.")

# if __name__ == "__main__":
#     main()