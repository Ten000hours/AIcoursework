import run_simple
import run_rl

import matplotlib.pyplot as plot


def enviornment_select(type="run_rl"):
    if type == "run_rl":
        env, problem_id, epsilon, total_episodes, max_steps, lr_rate, gamma, Q = run_rl.environment_eval(1)
        return env, problem_id, epsilon, total_episodes, max_steps, lr_rate, gamma, Q
    elif type == "run_simple":
        maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos = run_simple.enviornment_eval(
            1)
        return maze_map, problem_id, state_space_locations, state_space_actions, state_initial_id, state_goal_id, initial_node_colors, node_label_pos


def running(env, epsilon, total_episodes, max_steps, lr_rate, gamma, Q):
    reward_list, iter_list = run_rl.main(total_episodes, env, max_steps, Q, gamma, lr_rate, epsilon)
    return reward_list, iter_list


def plot_reward(reward_list):
    plot.plot(reward_list)
    plot.show()


def plot_iter(iter_list):
    plot.plot(iter_list)
    plot.show()


if __name__ == '__main__':
    env, problem_id, epsilon, total_episodes, max_steps, lr_rate, gamma, Q = enviornment_select("run_rl")
    reward_list, iter_list = running(env, epsilon, total_episodes, max_steps, lr_rate, gamma, Q)
    plot_iter(iter_list)
