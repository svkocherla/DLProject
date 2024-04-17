import numpy as np

class QLearning:
    def __init__(self, model, max_episodes):
        self.model = model
        self.max_episodes = max_episodes

    def train(self, env, verbose = True, shuffle_cap = None, max_steps = 100):
        for episode in range(self.max_episodes):
            env.reset()
            shuffle = shuffle_cap if episode // 1000 + 1 > shuffle_cap else episode // 1000 + 1
            while env.is_solved():
                env.shuffle_n(shuffle)

            state = env.get_state()
            done = False
            steps = 0

            while not done and steps < max_steps:
                steps += 1
                action = self.model.train_action(state, eps = 0.9 / (np.log(episode + 2)))
                reward = env.process_action(action)
                next_state = env.get_state()
                self.model.update(state, action, reward, next_state)
                state = next_state
                done = env.is_solved()

            if verbose:
                if episode % 1000 == 0:
                    print(f"Training Episode: {episode} with shuffle {shuffle}")
                
        print("Completed Training")

    def test(self, env, step_limit = 100, max_shuffle = 20, verbose = True, very_verbose = False, set_shuffle = False):
        env.reset()
        done = False
        total_reward = 0
        steps = 0

        par = np.random.randint(1, max_shuffle) if not set_shuffle else max_shuffle
        while env.is_solved():
            env.shuffle_n(par)
        state = env.get_state()
        
        if very_verbose:
            print("Initial Grid:")
            env.print_grid()

        while not done and steps < step_limit:
            steps += 1
            action = self.model.test_action(state)
            reward = env.process_action(action)
            state = env.get_state()
            total_reward += reward
            done = env.is_solved()

        if steps < step_limit:
            if verbose:
                tmp = "worse" if steps - par > 0 else "better"
                print(f"Completed in {steps} steps, {steps - par if steps - par > 0 else par - steps} steps {tmp} than par of {par}, Total Reward: {total_reward}")
            return steps
        else:
            if verbose:
                print(f"Timed out in {steps} steps")
            return -1

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
