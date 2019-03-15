import run_simple
import run_rl
import run_random
import sys
import matplotlib.pyplot as plot


def enviornment_select(type="run_rl"):
    if type == "run_rl":
        id = int(sys.argv[1])
        env, problem_id, epsilon, total_episodes, max_steps, lr_rate, gamma, Q = run_rl.environment_eval(id)
        return env, problem_id, epsilon, total_episodes, max_steps, lr_rate, gamma, Q
    elif type == "run_simple":
        id = int(sys.argv[1])
        maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos = run_simple.enviornment_eval(
            id)
        return maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos
    elif type== "run_random":
        id= int(sys.argv[1])
        max_episodes,env,max_iter_per_episode,observation_list,reward_list=run_random.eviornment()
        return max_episodes,env,max_iter_per_episode,observation_list,reward_list

def running_rl(env, epsilon, total_episodes, max_steps, lr_rate, gamma, Q):
    reward_list, iter_list = run_rl.main(total_episodes, env, max_steps, Q, gamma, lr_rate, epsilon)
    return reward_list, iter_list


def running_simple(state_initial_id, maze_problem, initial_node_colors):
    solution_path, iterations = run_simple.main(state_initial_id, maze_problem, initial_node_colors)
    return solution_path, iterations

def running_random(max_episodes,env,max_iter_per_episode,observation_list,reward_list):
    observation_list,reward_list=run_random.main(max_episodes,env,max_iter_per_episode,0.0,observation_list,reward_list)
    return observation_list,reward_list

def plot_reward(reward_list):
    plot.plot(reward_list)
    plot.show()


def plot_iter(iter_list):
    plot.plot(iter_list)
    plot.show()


if __name__ == '__main__':
    env, problem_id, epsilon, total_episodes, max_steps, lr_rate, gamma, Q = enviornment_select("run_rl")
    reward_list, iter_list = running_rl(env, epsilon, total_episodes, max_steps, lr_rate, gamma, Q)
    plot.plot(iter_list)

    #
    maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos = enviornment_select(
        "run_simple")
    maze_problem = run_simple.mazeproblem(state_initial_id, state_goal_id, maze_map)
    solution_path, iterations = running_simple(state_initial_id, maze_problem, initial_node_colors)
    plot.axhline(y=iterations, xmin=0, xmax=200, color='k')

    #
    max_episodes,env,max_iter_per_episode,observation_list,reward_list_random=enviornment_select("run_random")
    observation_list,reward_list_random=running_random(max_episodes,env,max_iter_per_episode,observation_list,reward_list_random)
    plot.plot(y=observation_list,color="y")

    plot.title("how many iterations for each agent when sucess")
    plot.xlabel("episode")
    plot.ylabel("iterations")
    plot.legend(["qlearning iter","simple iter","random iter"],loc="upper left")
    plot.savefig("iter_"+str(problem_id)+".png")
    plot.show()

