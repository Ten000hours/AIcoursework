# this is the agent of random to take random move for the problem
# this is the warpper of randomState and randint method to take  the random move
import gym
from uofgsocsai import *

class runRandom:
    def action(self):
        # take random action
        env = LochLomondEnv(problem_id=0, is_stochastic=True, reward_hole=0)

        return env.action_space.sample()
