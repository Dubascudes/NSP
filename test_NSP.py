import re
import openai
import json
import networkx as nx
import time
import tiktoken
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import partial
import itertools
from itertools import permutations
import threading
import os

timeout_duration_local = 6
timeout_duration_global = 15*50 + 1



class TimeoutException(Exception):
    pass

openai.api_key = os.environ("OPENAI_API_KEY")

encoding = tiktoken.encoding_for_model("gpt-4")
class TextColor:
    RED = "\033[91m"
    GREEN = "\033[92m"
    ORANGE = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

def extract_functions(generated_code):
    # Extract all Python code blocks
    code_blocks = []
    start_index = 0
    while start_index != -1:
        start_index = generated_code.find("```python", start_index)
        if start_index == -1:
            break
        end_index = generated_code.find("```", start_index + len("```python"))
        if end_index == -1:
            break
        code_block = generated_code[start_index + len("```python"):end_index].strip()
        code_blocks.append(code_block)
        start_index = end_index + len("```")
    generated_code = "\n\n".join(code_blocks)
    return generated_code

def map_path_labels(path, graph):
    # Create a mapping dictionary between graph labels and path labels
    mapping = {}
    graph_labels = list(graph.nodes())
    
    for node in path:
        # Convert integer node to its corresponding string label
        label = f"Room{node}"
        if label in graph_labels:
            mapping[node] = label
        else:
            return None  # If no matching label is found, the path is invalid
    
    # Map the path labels to the graph labels
    mapped_path = [mapping[node] for node in path]
    return mapped_path

def is_path_legal(path, ground_truth_path, graph):
    if path is None or len(path) < 2:
        return False
    
    # Check if the path and graph have the same node label type
    if not all(isinstance(node, type(list(graph.nodes())[0])) for node in path):
        path = map_path_labels(path, graph)


    if path[0] == ground_truth_path[0] and path[-1] == ground_truth_path[-1]:
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i + 1]):
                return False
    return True
    
def self_verify(path, graph):
    if path is None or path is [] or len(path) < 2:
        return False
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True

def construct_graph(graph_entry):
    G = nx.Graph()
    if "connections" in graph_entry:
        for node, neighbors in graph_entry["connections"].items():
            if isinstance(neighbors, list):
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            else:
                for neighbor, weight in neighbors.items():
                    G.add_edge(node, neighbor, weight=weight)
    elif "distance_graph" in graph_entry:
        for node, neighbors in graph_entry["distance_graph"].items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)
    else:
        raise ValueError("Invalid graph entry. Must contain 'connections' or 'distance_graph' key.")
    return G


def normalize_graph(graph):
    # Check the type of the first node's label
    first_node = list(graph.nodes())[0]
    
    if isinstance(first_node, int):
        # Relabel nodes to strings of the form "RoomX"
        mapping = {node: f"Room{node}" for node in graph.nodes()}
        graph = nx.relabel_nodes(graph, mapping)
        
    return graph

def normalize_path(path):
    # Check the type of the first node's label in the graph    
    if not path: return []
    if isinstance(path[0], int):
        # Relabel path nodes to strings of the form "RoomX"
        path = [f"Room{node}" for node in path]
        
    return path

def is_tsp_success(path, graph, start):
    # Normalize the graph and path
    graph = normalize_graph(graph)
    path = normalize_path(path)

    # Check if the path starts and ends at the starting node
    if path[0] != start or path[-1] != start:
        print("Invalid start node")
        return False
    
    # Check if the path visits all nodes
    if len(path) < graph.number_of_nodes()+1:
        print(f"doesn't visit all nodes, path length {len(path)}, graph size {graph.number_of_nodes()}")
        return False
    
    # Check if all nodes in the graph are visited
    for node in graph.nodes():
        if node not in path:
            return False
    
    return True

def construct_prompt(description, function_code=None, error_message=None):
    prompt = f"""
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
"""
    if error_message:
        prompt += f"\n\nAn error occurred with the previous response: \n{function_code}\n\n The error message was: \n{error_message}\nPlease correct the response. You may have misunderstood the problem or the type of graph to use in this situation."
    
    return prompt
def run_exec_with_timeout(function_code, local_vars, timeout):
    def target():
        try:
            exec(function_code, globals(), local_vars)
        except Exception as e:
            local_vars["exec_error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutException("Execution timed out")
    if "exec_error" in local_vars:
        raise local_vars["exec_error"]
import gc
import tracemalloc

# Start tracing memory allocations
tracemalloc.start()

def handle_trial(trial, is_tsp, attempts):
    description = trial["entry"]["description"]
    if trial["entry"].get("constraints"):
        description += " " + trial["entry"]["constraints"]
    prompt = construct_prompt(description, trial["function_code"], trial["error_message"])
    
    start_time = time.time()
    
    try:
        # signal.alarm(300)  # Set an alarm for 300 seconds
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        # signal.alarm(0)  # Disable the alarm
        
        end_time = time.time()
        inference_time = end_time - start_time
        generated_code = completion.choices[0].message.content.strip()
        output_tokens = len(encoding.encode(generated_code))
        total_tokens = len(encoding.encode(prompt)) + output_tokens
        function_code = extract_functions(generated_code)
        trial["function_code"] = function_code

        try:
            local_vars = {}
            run_exec_with_timeout(function_code, local_vars, timeout=4)  # 5-minute timeout
            
            house_graph = local_vars["create_graph"]()
            generated_path = local_vars["solve_problem"](house_graph, local_vars["args"])

            ground_truth = trial["entry"].get("ground_truth") or trial["entry"].get("cycle")
            normed_generated_graph = normalize_graph(house_graph)
            normed_generated_path = normalize_path(generated_path)
            is_self_legal = self_verify(normed_generated_path, normed_generated_graph)
            efficiency = 0
            is_legal = False
            is_optimal = False
            is_success = False
            if is_self_legal:
                if is_tsp:
                    is_legal = is_path_legal(ground_truth, generated_path, construct_graph(trial["entry"]))
                    is_success = is_tsp_success(normed_generated_path, normed_generated_graph, ground_truth[0])
                else:
                    is_legal = is_path_legal(generated_path, ground_truth, construct_graph(trial["entry"]))

                if ground_truth and is_legal and is_success:
                    is_optimal = (is_legal and len(generated_path) == len(ground_truth))
                    efficiency = len(generated_path) / len(ground_truth)

            else:
                trial["error_message"] += f"The path {str(generated_path)} is not legal for the graph."

            result_entry = {
                "num_rooms": trial.get("num_rooms"),
                "perturbed": trial.get("perturbed"),
                "description": description,
                "ground_truth_path": ground_truth,
                "generated_path": generated_path,
                "ground_truth_graph": trial["entry"].get("connections") or trial["entry"].get("distance_graph"),
                "generated_graph": {node: list(neighbors) for node, neighbors in house_graph.adjacency()},
                "function_code": function_code,
                "success": is_success,
                "self_success": is_self_legal,
                "legal_path": is_legal,
                "efficiency": efficiency,
                "optimal_path": is_optimal,
                "attempts": attempts,
                "inference_time": inference_time,
                "token_count": total_tokens
            }
            return {
                "trial": trial,
                "success": is_legal,
                "self_success": is_self_legal,
                "legal_path": is_legal,
                "optimal_path": is_optimal,
                "result_entry": result_entry
            }
        except TimeoutException:
            generated_path = None
            trial["error_message"] = "The code took too long to execute."
            trial["attempts"] += 1
            result_entry = {
                "num_rooms": trial.get("num_rooms"),
                "perturbed": trial.get("perturbed"),
                "description": description,
                "ground_truth_path": trial["entry"].get("ground_truth") or trial["entry"].get("cycle"),
                "generated_path": None,
                "ground_truth_graph": trial["entry"].get("connections") or trial["entry"].get("distance_graph"),
                "generated_graph": None,
                "function_code": function_code,
                "success": False,
                "self_success": False,
                "legal_path": False,
                "optimal_path": False,
                "error_message": trial["error_message"],
                "attempts": attempts,
                "inference_time": None,
                "token_count": total_tokens
            }
            return {
                "trial": trial,
                "success": False,
                "self_success": False,
                "legal_path": False,
                "optimal_path": False,
                "error_message": trial["error_message"],
                "attempts": trial["attempts"],
                "result_entry": result_entry
            }
    except TimeoutException:
        trial["error_message"] = "The code took too long to execute."
        trial["attempts"] += 1
        result_entry = {
            "num_rooms": trial.get("num_rooms"),
            "perturbed": trial.get("perturbed"),
            "description": description,
            "ground_truth_path": trial["entry"].get("ground_truth") or trial["entry"].get("cycle"),
            "generated_path": None,
            "ground_truth_graph": trial["entry"].get("connections") or trial["entry"].get("distance_graph"),
            "generated_graph": None,
            "function_code": function_code,
            "success": False,
            "self_success": False,
            "legal_path": False,
            "optimal_path": False,
            "error_message": trial["error_message"],
            "attempts": attempts,
            "inference_time": None,
            "token_count": total_tokens
        }
        return {
            "trial": trial,
            "success": False,
            "self_success": False,
            "legal_path": False,
            "optimal_path": False,
            "error_message": trial["error_message"],
            "attempts": trial["attempts"],
            "result_entry": result_entry
        }
    except Exception as e:
        trial["error_message"] = str(traceback.format_exc())
        trial["attempts"] += 1
        result_entry = {
            "num_rooms": trial.get("num_rooms"),
            "perturbed": trial.get("perturbed"),
            "description": description,
            "ground_truth_path": trial["entry"].get("ground_truth") or trial["entry"].get("cycle"),
            "generated_path": None,
            "ground_truth_graph": trial["entry"].get("connections") or trial["entry"].get("distance_graph"),
            "generated_graph": None,
            "function_code": function_code,
            "success": False,
            "self_success": False,
            "legal_path": False,
            "optimal_path": False,
            "error_message": trial["error_message"],
            "attempts": attempts,
            "inference_time": None,
            "token_count": total_tokens
        }
        return {
            "trial": trial,
            "success": False,
            "self_success": False,
            "legal_path": False,
            "optimal_path": False,
            "error_message": trial["error_message"],
            "attempts": trial["attempts"],
            "result_entry": result_entry
        }
    finally:
        # Force garbage collection
        gc.collect()
        # Print memory usage
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

# Your existing test_llm_navigation function modified
def test_llm_navigation(dataset_name, is_tsp, test_dataset, num_entries=50, start_points=None):
    if start_points is None:
        if is_tsp:
            start_points = {num_rooms: 0 for num_rooms in [5, 10, 15, 20, 25]}
        else:
            start_points = {key: 0 for key in [(num_rooms, perturbed) for num_rooms in [5, 10, 15, 20, 25] for perturbed in [True, False]]}

    results = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "legal_path": 0,
        "optimal_path": 0,
        "details": []
    }

    room_results = {5: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                    10: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                    15: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                    20: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                    25: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0}}

    if is_tsp:
        room_results = {5: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                        10: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                        15: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                        20: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                        25: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0}}

    perturbed_results = {True: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                         False: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0}}

    filtered_datasets = {}
    if is_tsp:
        for num_rooms in room_results.keys():
            filtered_datasets[num_rooms] = [entry for entry in test_dataset if entry['num_rooms'] == num_rooms]
    else:
        for num_rooms in room_results.keys():
            for perturbed in perturbed_results.keys():
                filtered_datasets[(num_rooms, perturbed)] = [entry for entry in test_dataset 
                                                             if entry['num_rooms'] == num_rooms 
                                                             and entry['perturbed'] == perturbed]

    # For each group of num_rooms in our dataset
    for key, dataset in filtered_datasets.items():
        if len(dataset) < num_entries:
            print(f"Warning: Not enough data for num_rooms={key}. Available: {len(dataset)}")
        if is_tsp:
            num_rooms = key
            perturbed = None
            start_index = start_points[num_rooms]
        else:
            num_rooms, perturbed = key
            start_index = start_points[(num_rooms, perturbed)]

        trials = [{"entry": entry, "attempts": 0, "function_code": None, "error_message": None, "num_rooms": num_rooms, "perturbed": perturbed} for entry in dataset[start_index:num_entries]]
        attempts = 1
        while len(trials) != 0 and attempts <= 5:
            print(f"{dataset_name} {key} Attempt #{attempts}: {len(trials)} trials")

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_results = [executor.submit(handle_trial, trial, is_tsp, attempts) for trial in trials]
                batch_results = []
                print("Fetching future results...")
                for future, trial in zip(future_results, trials):
                    print(f"Batch result #{len(batch_results)}")
                    try:
                        batch_results.append(future.result(timeout=4))  # Timeout of 5 minutes
                    except TimeoutException:
                        print(f"TimeoutError: Trial for {trial['entry']['description']} took too long and was skipped.")
                        trial["error_message"] = "The code took too long to execute."
                    except Exception as e:
                        print(f"Exception: {e}")
                        trial["error_message"] = str(e)
                    finally:
                        future.cancel()
            print(f"{TextColor.BLUE}Updating trial results...{TextColor.RESET}")
            new_trials = []
            for trial, result in zip(trials, batch_results):
                trial_data = result["trial"]
                if not result["self_success"] and attempts < 5:
                    new_trials.append(trial_data)
                    results["total"] -= 1
                    results["failure"] -= 1
                else:
                    existing_entry = next((entry for entry in results["details"] if entry["description"] == trial_data["entry"]["description"] and entry["attempts"] == trial_data["attempts"] - 1), None)
                    if existing_entry and attempts < 5:
                        results["details"].remove(existing_entry)
                    results["details"].append(result["result_entry"])

                results["total"] += 1
                if result["success"]:
                    results["success"] += 1
                    results["legal_path"] += result["legal_path"]
                    results["optimal_path"] += result["optimal_path"]
                else:
                    results["failure"] += 1

            attempts += 1
            trials = new_trials
            print(f"{len(new_trials)} trials failed.")

            with open(f"navigation_results_{dataset_name}_NSP_batch.json", "w") as f:
                json.dump(results, f, indent=4)
            with open(f"room_results_{dataset_name}_NSP_batch.json", "w") as f:
                json.dump(room_results, f, indent=4)

    return results, room_results




def process_datasets(datasets, num_entries=50):
    for dataset_name in datasets:
        with open(f"{dataset_name}.json", "r") as f:
            test_dataset = json.load(f)
            is_tsp = "tsp" in dataset_name.lower()

            results, room_results = test_llm_navigation(dataset_name, is_tsp, test_dataset, num_entries=num_entries, start_points=None)
            
            print(f"Results for {dataset_name} with NSP:")
            print(f"Total: {results['total']}, Success: {results['success']}, Failure: {results['failure']}")
            print(f"Legal Paths: {results['legal_path']}, Optimal Paths: {results['optimal_path']}")

            with open(f"navigation_results_{dataset_name}_NSP_batch.json", "w") as f:
                json.dump(results, f, indent=4)
            with open(f"room_results_{dataset_name}_NSP_batch.json", "w") as f:
                json.dump(room_results, f, indent=4)

if __name__ == "__main__":
    datasets = ["cleaned_dataset_unweighted_tsp", "cleaned_dataset_weighted_tsp", "cleaned_dataset_unweighted_paths", "cleaned_dataset_weighted_paths"]
    process_datasets(datasets, num_entries=50)

#"test_dataset_paths_unweighted",
#"cleaned_dataset_unweighted_tsp", "cleaned_dataset_weighted_tsp", 