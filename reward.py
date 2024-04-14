import numpy as np
def naive(self, old_state, action, new_state):
    # return reward from (old state, action) -> (new state, reward)
    if np.all(old_state == new_state):
        return -1
    if self.is_solved():
        return 100
    return 0