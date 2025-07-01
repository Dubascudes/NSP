# NSP: A Neuro-Symbolic Natural Language Navigational Planner

This repository contains materials related to (this paper)[https://ieeexplore.ieee.org/abstract/document/10903323] published at ICMLA 2024. A preprint version can be found (here)[https://arxiv.org/pdf/2409.06859]. In our paper, we comapre our proposed framework against 4 other prompt-based approaches. Ours is distinguished by the inclusion of a solver with which the reasoning model can interact with. This solver is the Python interpreter + networkx, a library for reasoning about graphs.


# Instructions
Each of the `test_xxx.py` files requires an environemnt variable `OPENAI_API_KEY` to be set. 
We primarily test two natural-language navigation problem types- shortest path and traveling salesman. Example datasets in both of these categories can be found in the `data` directory, or generated algorithmically using any of the scripts in `data_generators`.

# Our Framework

![NSP Framework Diagram, consisting of natural language inputs on the left, divided into environment description, specification, and constraints. These are fed into neuro-symbolic translation with an addition Graph Library API as input, which results in a graph, a candidate solution algorithm, and the relevant parameters for the algorithm as determined by the model. These are fed to the python interpreter, establishing a feedback loop between the interpreter and the LLM. The result is ultimately a formal NetworkX path on a graph.](imgs/framework_diagram.png)

We define the natural language path-planning problem as some NL string that includes a sufficient description of the environment, a path specification, and any constraints imposed on the path. To produce a solution to such a problem, we provide additional information in our prompt to the LLM, including information about the networkx library.

### The full NSP prompt is: 
```
Carefully consider the following path planning environment and scenario:
"{description}"

Write a Python function create_graph() that generates an undirected weighted or unweighted graph depending on the problem using the NetworkX library based on the path planning problem. The function should return a NetworkX weighted graph object.

Additionally, write another function solve_problem(graph, args) that solves the path planning problem in the form of a node traversal order list.

Your code must comply with these guidelines:

Your response must include the complete function code and a defined instance of args, which is an array containing any arguments needed for the solve_problem function.
Do not return any incomplete functions or comment out any of the functions or definitions.
Do not call or test the code.
Nested function definitions are not allowed. All functions must be global.
The available libraries are networkx and itertools.
Guidelines for solution efficiency:

If the problem is similar to another problem with a known efficient solution involving techniques such as dynamic programming, use it in your implementation.
Your solve_problem() function should return the shortest path that satisfies the objective of the path planning scenario.
You may approximate the solution as a fallback if you expect an algorithm would take more than a minute to execute. 
Here are some helpful NetworkX methods you can use. For each of the following, the argument types are G: NetworkX Graph, source: node, target: node, weight: string, cycle: bool, and method: function:

networkx.shortest_path(G, source, target): Finds the shortest path from a source node to a target node.
networkx.dijkstra_path(G, source, target): Finds the shortest path using Dijkstra's algorithm.
traveling_salesman_problem(G[, weight, ...]): Finds the shortest path in G connecting specified nodes.
greedy_tsp(G[, weight, source]): Returns a low-cost cycle starting at source and its cost.
simulated_annealing_tsp(G, init_cycle[, ...]): Returns an approximate solution to the traveling salesman problem.
threshold_accepting_tsp(G, init_cycle[, ...]): Returns an approximate solution to the traveling salesman problem.
asadpour_atsp(G[, weight, seed, source]): Returns an approximate solution to the traveling salesman problem.
networkx.minimum_spanning_tree(G): Finds the minimum spanning tree of the graph, which is useful in some path planning scenarios.
networkx.connected_components(G): Finds all connected components in the graph, which is useful for understanding the reachability within the graph.
networkx.find_cycle(G): Finds a cycle in the graph, useful for detecting loops in the paths.
```

There are **four** main components of this prompt:

1. Problem Description
2. `create_graph()`
3. `solve_problem(graph, args)`
4. NetworkX API information

The idea is that the LLM will return `python` code blocks that contain our two functions `create_graph()` and `solve_problem(graph, args)`. We parse these blocks using the `extract_functions()` function in `test_NSP.py`, and feed these to the Python interpreter. We execute the LLM-generated code in the `try` block on lines 235-289 in `test_NSP.py`. 


## Citation
W. English, D. Simon, S. K. Jha and R. Ewetz, "NSP: A Neuro-Symbolic Natural Language Navigational Planner," 2024 International Conference on Machine Learning and Applications (ICMLA), Miami, FL, USA, 2024, pp. 1289-1294, doi: 10.1109/ICMLA61862.2024.00201. keywords: {Feedback loop;Navigation;Natural languages;Syntactics;Programming;Path planning;Planning;Time factors;Robots;Python;Neuro-Symbolic AI;Navigation;LLM;Spatial Reasoning;Path Planning},

