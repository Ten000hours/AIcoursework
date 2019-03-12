from uofgsocsai import *
import gym
import networkx as nx
print("networkx version:"+nx.__version__)

from search import *
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display


from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import lines

from helpers import *

problem_id=0
# first, parse the env.desc to undirected map using aima tool box
env= LochLomondEnv(problem_id=problem_id,is_stochastic=False,reward_hole=0.0)

print(env.desc)

state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

# print(state_space_actions)

maze_map= UndirectedGraph(state_space_actions)
# print(maze_map.nodes())
#
# maze_map = UndirectedGraph(dict(
#     S_00_00=dict(S_01_00=1,S_00_00=1),
#     S_01_00=dict(S_01_01=1,S_02_00=1),
#     S_02_00=dict(S_02_01=1,S_03_00=1),
#     S_03_00=dict(S_03_01=1,S_04_00=1),
#     S_04_00=dict(S_04_01=1),
#     S_06_00=dict(S_06_01=1,S_07_00=1),
#     S_07_00=dict(S_07_01=1),
#     S_00_01=dict(S_00_02=1,S_01_01=1),
#     S_01_01=dict(S_01_02=1,S_02_01=1),
#     S_02_01=dict(S_03_01=1,S_02_02=1),
#     S_03_01=dict(S_04_01=1),
#     S_04_01=dict(S_04_02=1,S_05_01=1),
#     S_05_01=dict(S_05_02=1,S_06_01=1),
#     S_06_01=dict(S_06_02=1,S_07_01=1),
#     S_07_01=dict(S_07_02=1),
#     S_00_02=dict(S_00_03=1,S_01_02=1),
#     S_01_02=dict(S_01_03=1,S_02_02=1),
#     S_02_02=dict(S_02_03=1),
#
#     S_04_02=dict(S_04_03=1,S_05_02=1),
#     S_05_02=dict(S_06_02=1),
#     S_06_02=dict(S_06_03=1,S_07_02=1),
#     S_07_02=dict(S_07_03=1),
#     S_00_03=dict(S_00_04=1,S_01_03=1),
#     S_01_03=dict(S_01_04=1,S_02_03=1),
#     S_02_03=dict(S_02_04=1,S_03_03=1),
#     S_03_03=dict(S_04_03=1),
#     S_04_03=dict(S_04_04=1),
#     S_06_03=dict(S_06_04=1,S_07_03=1),
#     S_07_03=dict(S_07_04=1),
#     S_00_04=dict(S_01_04=1,S_00_05=1),
#     S_01_04=dict(S_02_04=1),
#
#     S_04_04=dict(S_05_04=1,S_04_05=1),
#     S_05_04=dict(S_05_05=1,S_06_04=1),
#     S_06_04=dict(S_07_04=1),
#     S_07_04=dict(S_07_05=1),
#     S_00_05=dict(S_00_06=1),
#     S_03_05=dict(S_04_05=1,S_03_06=1),
#     S_04_05=dict(S_05_05=1),
#     S_05_05=dict(S_05_06=1),
#     S_07_05=dict(S_07_06=1),
#     S_00_06=dict(S_00_07=1),
#     S_02_06=dict(S_02_07=1,S_03_06=1),
#     S_05_06=dict(S_05_07=1),
#     S_07_06=dict(S_07_07=1),
#     S_00_07=dict(S_01_07=1),
#     S_01_07=dict(S_02_07=1),
#     S_04_07=dict(S_05_07=1),
#     S_05_07=dict(S_06_07=1),
#     # S_09_07=dict(S_08_07=1,S_09_06=1),
#     # S_00_08=dict(S_00_09=1,S_01_08=1,S_00_07=1),
#     # S_01_08=dict(S_00_08=1,S_02_08=1,S_01_09=1,S_01_07=1),
#     # S_02_08=dict(S_02_09=1,S_02_07=1,S_01_08=1),
#     # S_04_08=dict(S_04_09=1,S_05_08=1,S_04_07=1),
#     # S_05_08=dict(S_04_08=1,S_06_08=1,S_05_09=1,S_05_07=1),
#     # S_06_08=dict(S_07_08=1,S_06_07=1,S_05_08=1),
#     # S_07_08=dict(S_07_09=1,S_06_08=1),
#     # S_01_09=dict(S_01_08=1),
#     # S_00_09=dict(S_01_09=1,S_00_08=1),
#     # S_02_09=dict(S_01_09=1,S_02_08=1),
#     # S_04_09=dict(S_04_08=1,S_05_09=1),
#     # S_05_09=dict(S_05_08=1,S_04_09=1),
#     # S_07_09=dict(S_07_08=1))
# ))
#
# maze_map.locations = dict(
#     S_00_00=(0,0),
#     S_00_03=(0,3),
#     S_00_04=(0,4),
#     S_01_00=(1,0),
#     S_02_00=(2,0),
#     S_03_00=(3,0),
#     S_04_00=(4,0),
#     S_06_00=(6,0),
#     S_07_00=(7,0),
#     S_00_01=(0,1),
#     S_01_01=(1,1),
#     S_02_01=(2,1),
#     S_03_01=(3,1),
#     S_04_01=(4,1),
#     S_05_01=(5,1),
#     S_06_01=(6,1),
#     S_07_01=(7,1),
#     S_00_02=(0,2),
#     S_01_02=(1,2),
#     S_02_02=(2,2),
#     S_04_02=(4,2),
#     S_05_02=(5,2),
#     S_06_02=(6,2),
#     S_07_02=(7,2),
#     S_01_03=(1,3),
#     S_02_03=(2,3),
#     S_03_03=(3,3),
#     S_04_03=(4,3),
#
#     S_06_03=(6,3),
#     S_07_03=(7,3),
#     S_01_04=(1,4),
#     S_02_04=(2,4),
#     S_04_04=(4,4),
#     S_05_04=(5,4),
#     S_06_04=(6,4),
#     S_07_04=(7,4),
#
#     S_00_05=(0,5),
#
#     S_03_05=(3,5),
#     S_04_05=(4,5),
#     S_05_05=(5,5),
#     S_07_05=(7,5),
#
#     S_00_06=(0,6),
#     S_03_06=(3,6),
#
#     S_02_06=(2,6),
#
#     S_05_06=(5,6),
#
#     S_07_06=(7,6),
#
#     S_00_07=(0,7),
#
#     S_01_07=(1,7),
#     S_06_07=(6,7),
#     S_02_07=(2,7),
#     S_05_07=(5,7),
#     S_04_07=(4,7),
#     S_07_07=(7,7)
#
#     )

# maze_map_locations = maze_map.locations
#
# print("----------------------------")
# print("Unique states (i.e.locations) and possible actions in those states:\n")
# print(maze_map)

# initialise a graph
G = nx.Graph()

# use this while labeling nodes in the map
node_labels = dict()
node_colors = dict()
for n, p in state_space_locations.items():  #maze_map_locations
    G.add_node(n)  # add nodes from locations
    node_labels[n] = n  # add nodes to node_labels
    node_colors[n] = "white"  # node_colors to color nodes while exploring the map

# we'll save the initial node colors to a dict for later use
initial_node_colors = dict(node_colors)

# positions for node labels
node_label_pos = {k: [v[0], v[1] - 0.25] for k, v in
                  state_space_locations.items()}  # spec the position of the labels relative to the nodes         maze_map_locations

# use this while labeling edges
edge_labels = dict()

# add edges between nodes in the map - UndirectedGraph defined in search.py
for node in maze_map.nodes():
    connections = maze_map.get(node)
    for connection in connections.keys():
        distance = connections[connection]
        G.add_edge(node, connection)  # add edges to the graph
        edge_labels[(node, connection)] = distance  # add distances to edge_labels

print("Done creating the graph object")


def show_map(node_colors):
    # set the size of the plot
    plt.figure(figsize=(16, 13))
    # draw the graph (both nodes and edges) with locations
    nx.draw_networkx(G, pos=state_space_locations, node_color=[node_colors[node] for node in G.nodes()])    # maze_map_locations

    # draw labels for nodes
    node_label_handles = nx.draw_networkx_labels(G, pos=node_label_pos, labels=node_labels, font_size=9)
    # add a white bounding box behind the node labels
    [label.set_bbox(dict(facecolor='white', edgecolor='none')) for label in node_label_handles.values()]

    # add edge lables to the graph
    nx.draw_networkx_edge_labels(G, pos=state_space_locations, edge_labels=edge_labels, font_size=8)    # maze_map_locations

    # add a legend
    white_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="white")
    orange_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="orange")
    red_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="red")
    gray_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="gray")
    green_circle = lines.Line2D([], [], color="white", marker='o', markersize=15, markerfacecolor="green")
    plt.legend((white_circle, orange_circle, red_circle, gray_circle, green_circle),
               ('Un-explored', 'Frontier', 'Currently exploring', 'Explored', 'Solution path'),
               numpoints=1, prop={'size': 16}, loc=(.8, 1.0))




maze_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)

print("Initial state: " + maze_problem.initial)
print("Goal state: "    + maze_problem.goal)


def my_best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = dict(initial_node_colors)

    f = memoize(f, 'f')
    node = Node(problem.initial)

    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return (iterations, all_node_colors, node)

    frontier = PriorityQueue('min', f)
    frontier.append(node)

    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    explored = set()
    while frontier:
        node = frontier.pop()

        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

def my_astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h') # define the heuristic function
    return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def final_path_colors(problem, solution):
    "returns a node_colors dict of the final path provided the problem and solution"

    # get initial node colors
    final_colors = dict(initial_node_colors)
    # color all the nodes in solution and starting node to green
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"
    return final_colors


def display_visual(user_input, algorithm=None, problem=None):
    if user_input == False:
        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(all_node_colors[iteration])
            except:
                pass

        def visualize_callback(Visualize):
            if Visualize is True:
                button.value = False

                global all_node_colors

                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(final_path_colors(problem, solution))

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i
                    # time.sleep(3.)

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, Visualize=button)
        display(button_visual)


all_node_colors=[]
iterations, all_node_colors, node = my_astar_search(problem=maze_problem, h=None)
print(iterations)
#-- Trace the solution --#
solution_path = [node]
cnode = node.parent
solution_path.append(cnode)
while cnode.state != state_initial_id:
    cnode = cnode.parent
    solution_path.append(cnode)

print("----------------------------------------")
print("Identified goal state:"+str(solution_path[0]))
print("----------------------------------------")
print("Solution trace:"+str(solution_path))
print("----------------------------------------")
print("Final solution path:")
# show_map(final_path_colors(maze_problem, node.solution()))



filename= "out_simple_"+str(problem_id)
text="Identified goal state:"+str(solution_path[0])+"\n"+"Solution trace:"+str(solution_path)
print(str(text))
with open(filename+".txt", "w") as file:
    file.write(str(text))