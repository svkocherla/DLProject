import n_puzzle
import gymnasium
env = gymnasium.make('n_puzzle/n_puzzle-v0')

print(env.state)
env.reset(options={"k_moves": 10})
print(env.state)
print(env.step(0))
# for _ in range(5):
#     print(env.process_move(2))
#     print(env.state)