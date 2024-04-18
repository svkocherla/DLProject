import numpy as np
# import pygame

import gymnasium as gym
from gymnasium import spaces


class NPuzzleEnv(gym.Env):
    # TODO: implement rendering
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, n=5, k=10):
        self.n = n  # length of the square grid
        self.k = k  # default # of shuffles
        self.solved = np.arange(n*n).reshape(n,n)
        self.state = np.arange(n*n).reshape(n,n)
        self.emptyLocation = (n-1,n-1)
        # self.window_size = 512  # The size of the PyGame window

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

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        # """
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        # """
        # self.window = None
        # self.clock = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {"shuffle": self._shuffle}

    def generate_shuffle(self, k_moves: int) -> list[int]:
        moves = np.random.choice(4, k_moves)
        return moves
    
    # def clean_shuffle(self, moves: int) -> list[int]:
    #     #TODO: remove cycles and invalid moves
    #     pass

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
        moves = self.generate_shuffle(k_moves)
        valid = [self.process_move(move) for move in moves]
        self._shuffle = moves[valid]
    
    # def get_unshuffle(self, shuffle: list[int])-> list[int]:
    #     return np.flip((shuffle + 2) % 4)

    # def get_effective_k(self, shuffle: list[int]) -> int:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = self.solved.copy()
        k_moves = options.get("k_moves", self.k)
        self.shuffle(k_moves)

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

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

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info