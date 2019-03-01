import gym
import numpy as np
import time
import pickle, os
from uofgsocsai import *

env = LochLomondEnv(problem_id=1,is_stochastic=False,reward_hole=0.0)

with open("frozenLake_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)


def choose_action(state):
    action = np.argmax(Q[state, :])
    return action


# start
for episode in range(5):

    state = env.reset()
    print("*** Episode: ", episode)
    t = 0
    while t < 100:
        env.render()

        action = choose_action(state)

        state2, reward, done, info = env.step(action)

        state = state2

        if done:
            env.render()
            print("done")
            break

        time.sleep(1)
os.system('clear')