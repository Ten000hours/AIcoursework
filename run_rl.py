from time import sleep
import numpy as np
import gym
from uofgsocsai import *

# Environment
env = LochLomondEnv(problem_id=0,is_stochastic=True,reward_hole=0.0)
inputCount = env.observation_space.n
actionsCount = env.action_space.n

# Init Q-Table
Q = {}
for i in range(inputCount):
    Q[i] = np.random.rand(actionsCount)

# Hyperparameters
lr = 0.33
lrMin = 0.001
lrDecay = 0.9999
gamma = 1.0
epsilon = 1.0
epsilonMin = 0.001
epsilonDecay = 0.97
episodes = 2000

# Training
for i in range(episodes):
    print("Episode {}/{}".format(i + 1, episodes))
    s = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            a = np.random.randint(0, actionsCount)
        else:
            a = np.argmax(Q[s])

        newS, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[newS]) - Q[s][a])
        s = newS

        if lr > lrMin:
            lr *= lrDecay

        if not r==0 and epsilon > epsilonMin:
            epsilon *= epsilonDecay


print("")
print("Learning Rate :", lr)
print("Epsilon :", epsilon)
print(Q)
# Testing
print("\nPlay Game on 100 episodes...")

avg_r = 0
for i in range(100):
    s = env.reset()
    done = False

    while not done:
        a = np.argmax(Q[s])
        newS, r, done, _ = env.step(a)
        s = newS

    avg_r += r/100.

print("Average reward on 100 episodes :", avg_r)