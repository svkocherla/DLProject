import numpy as np

import gymnasium as gym
from gymnasium import spaces


class NPuzzleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config, n=5, k=1, render_mode=None,):
        self.n = env_config.get("n", n) # length of the square grid
        self.k = env_config.get("k", k)  # default # of shuffles

        self.solved = np.arange(n*n).reshape(n,n)
        self.state = np.arange(n*n).reshape(n,n)
        self.emptyLocation = (n-1,n-1)

        self.observation_space = spaces.Box(
            low=0,
            high=n**2 - 1,
            shape=(n, n),
            dtype=int
        )

        # We have 4 actions, corresponding to "UP", "DOWN", "LEFT", "RIGHT"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]), # UP
            1: np.array([1, 0]), # DOWN
            2: np.array([0, -1]), # LEFT
            3: np.array([0, 1]), # RIGHT
        }

        self._shuffle = []

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {"shuffle": self._shuffle}

    def generate_shuffle(self, k_moves: int) -> list[int]:
        moves = np.random.choice(4, k_moves)
        return moves
    
    def clean_shuffle(self, shuffle: list) -> list[int]:
        def is_undo(prev, curr):
            return abs(prev - curr) == 1 and min(prev, curr) != 1

        stack = []
        for x in shuffle:
            if not stack or not is_undo(stack[-1], x):
                stack.append(x)
            else:
                # stack is valid and move is an undo
                stack.pop()
        return np.array(stack, dtype=np.int64)
    

    def process_move(self, move: int) -> bool: 
        # moves empty location in direction of move
        change = self._action_to_direction[move]
        newLocation = self.emptyLocation + change
        if np.any((newLocation < 0) | (newLocation >= self.n)):
            return False
        
        newLocation = tuple(newLocation)
        self.state[self.emptyLocation], self.state[newLocation] = self.state[newLocation], self.state[self.emptyLocation]
        self.emptyLocation = newLocation
        return True
        
    def shuffle(self, k_moves: int):
        # generate (possibly invalid) shuffles
        moves = self.generate_shuffle(k_moves * 20)

        # remove invalid moves
        valid = [self.process_move(move) for move in moves]
        clean = self.clean_shuffle(moves[valid])

        # remove back and forths
        self._shuffle = clean[:k_moves]

        # reset board state
        self.state = self.solved.copy()
        self.emptyLocation = (self.n-1,self.n-1)

        # take the first_k valid moves
        valid = [self.process_move(move) for move in self._shuffle]
        self._shuffle = self._shuffle[valid]

    def get_unshuffle(self, shuffle: list[int])-> list[int]:
        if len(shuffle) == 0:
            return shuffle
        flip = lambda x: x ^ 0b1
        # return flip(np.flip(self.clean_shuffle(shuffle)))
        return flip(np.flip(shuffle))

    def get_effective_k(self, shuffle: list[int]) -> int:
        return len(self.clean_shuffle(shuffle))

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = self.solved.copy()
        self.emptyLocation = (self.n-1,self.n-1)

        k_moves = options.get("k_moves", self.k) if options else self.k
        self.shuffle(k_moves)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        #TODO: invalid moves
        valid = self.process_move(action)

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.state, self.solved)
        # TODO: function here
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info