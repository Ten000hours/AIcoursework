# this is the agent of random to take random move for the problem
# this is the warpper of randomState and randint method to take  the random move
import gym
from uofgsocsai import *

"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19

"""
import gym
import astar
import numpy as np
import time
import matplotlib.pyplot as plot

from uofgsocsai import LochLomondEnv  # load the class defining the custom Open AI Gym problem



class runRandom:
    def action(self):
        # take random action
        env = LochLomondEnv(problem_id=0, is_stochastic=True, reward_hole=0)

        return env.action_space.sample()


def eviornment():
    # Setup the parameters for the specific problem (you can change all of these if you want to)
    problem_id = int(sys.argv[1])# problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
    reward_hole = 0.0  # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

    max_episodes = 2000  # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_iter_per_episode = 500  # you decide how many iterations/actions can be executed per episode

    observation_list = list()
    reward_list = list()

    # Generate the specific problem
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)

    # Let's visualize the problem/env
    print('env', env.desc)

    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)
    return  max_episodes,env,max_iter_per_episode,observation_list,reward_list
def main(max_episodes,env,max_iter_per_episode,reward_hole,observation_list,reward_list):
    for e in range(max_episodes):  # iterate over episodes
        observation = env.reset()  # reset the state of the env to the starting state

        for iter in range(max_iter_per_episode):
            env.render()  # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line
            # action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
            random = runRandom()
            action = random.action()
            observation, reward, done, info = env.step(action)  # observe what happends when you take the action



            print("episode,iter,reward,done =" + str(e) + " " + str(iter) + " " + str(reward) + " " + str(done))

            # Check if we are done and monitor rewards etc...
            if (done and reward == reward_hole):
                env.render()
                print("We have reached a hole :-( [we can't move so stop trying; just give up]")
                break

            if (done and reward == +1.0):
                env.render()
                print(observation_list)
                observation_list.append(observation)
                reward_list.append(reward)
                print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
                break

    # print(observation_list)
    # plot.plot(observation_list)
    # plot.title("how many times that random agent success")
    # plot.xlabel("episodes")
    # plot.ylabel("iterations")
    # plot.show()
    return observation_list,reward_list
def filegenerate(problem_id,reward_list):
    filename = "out_random_" + str(problem_id)
    text = "rewards that obtained when the agent sucess to the goal:" + str(len(reward_list))+str(reward_list)
    print(str(text))
    with open(filename + ".txt", "w") as file:
        file.write(str(text))
if __name__ == '__main__':
    problem_id=int(sys.argv[1])
    max_episodes,env,max_iter_per_episode,observation_list,reward_list=eviornment()
    observation_list,reward_list=main(max_episodes,env,max_iter_per_episode,0.0,observation_list,reward_list)
    filegenerate(problem_id,reward_list)



