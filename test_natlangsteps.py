import re
import openai
import json
import networkx as nx
import matplotlib.pyplot as plt
import time  # Import the time module
import tiktoken  # Import tiktoken library for token counting
import os

openai.api_key = os.environ("OPENAI_API_KEY")

# Initialize the tokenizer for GPT-4
encoding = tiktoken.encoding_for_model("gpt-4")

# Function to generate step-by-step room transitions using GPT-4
def generate_transitions_from_description(description, constraints, error_message=None):
    prompt = f"""
    You are given a house with the following rooms and connections:
    "{description}"
    You need to find a path from the start room to the end room based on the following constraints:
    "{constraints}"
    Please provide the step-by-step room transitions, wrapped in ```. For example:
    ```
    Start in Room1
    Move to Room2
    Move to Room3
    End in Room4
    ```
    If there is no valid path, return:
    ```
    no valid path
    ```
    Do not use any code in your response.
    """
    if error_message:
        prompt += f"\n\nAn error occurred with the previous response: {error_message}\nPlease correct the response."

    # Calculate token count for the prompt
    input_tokens = len(encoding.encode(prompt))
    print(f"Input Token Count: {input_tokens}")

    messages = [
        {
            "role": "user",
            "content": prompt
        },
    ]

    start_time = time.time()  # Start timing
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end_time = time.time()  # End timing

    inference_time = end_time - start_time  # Calculate inference time

    generated_response = completion.choices[0].message.content.strip()
    print(generated_response)

    # Calculate token count for the output
    output_tokens = len(encoding.encode(generated_response))
    total_tokens = input_tokens + output_tokens
    print(f"Output Token Count: {output_tokens}")
    print(f"Total Token Count: {total_tokens}")

    # Parse the generated transitions
    # transitions_block = re.search(r"```(.*?)```", generated_response, re.DOTALL)
    # if not transitions_block:

        # return "no valid path", inference_time, total_tokens

    # transitions_text = transitions_block.group(1).strip()
    # if "no valid path" in transitions_text.lower():
        # return "no valid path", inference_time, total_tokens
    
    transitions = []
    for line in generated_response:
        match = re.match(r"Move to (\w+)|Start in (\w+)|End in (\w+)", line.strip())
        if match:
            transitions.append(match.group(1) or match.group(2) or match.group(3))
    if transitions[-2] == transitions[-1]: del transitions[-1]
    return transitions, inference_time, total_tokens

def is_path_legal(path, graph):
    if path == "no valid path":
        return False  # If the model correctly identifies no valid path, it's considered legal
    if path is None or len(path) < 2:
        return False
    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True

def is_path_optimal(path, ground_truth):
    if path == "no valid path" and ground_truth is None:
        return True
    if path == "no valid path" or ground_truth is None:
        return False
    return len(path) == len(ground_truth)

def test_llm_navigation(test_dataset, num_entries=100, start_points=None):
    print(start_points)
    if start_points is None:
        start_points = {key: 0 for key in [(num_rooms, perturbed) for num_rooms in [5, 10, 20, 30] for perturbed in [True, False]]}

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
                    20: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                    30: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0}}

    perturbed_results = {True: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0},
                         False: {"total": 0, "success": 0, "failure": 0, "legal_path": 0, "optimal_path": 0}}

    filtered_datasets = {}
    for num_rooms in room_results.keys():
        for perturbed in perturbed_results.keys():
                filtered_datasets[(num_rooms, perturbed)] = [entry for entry in test_dataset 
                                                             if entry['num_rooms'] == num_rooms 
                                                             and entry['perturbed'] == perturbed]

    for (num_rooms, perturbed), dataset in filtered_datasets.items():
        if len(dataset) < num_entries:
            print(f"Warning: Not enough data for num_rooms={num_rooms}, perturbed={perturbed}. Available: {len(dataset)}")

        start_index = start_points[(num_rooms, perturbed)]

        for idx, entry in enumerate(dataset[start_index:num_entries], start=start_index):
            description = entry["description"]
            constraints = entry["constraints"]
            ground_truth = entry["ground_truth"]
            ground_truth_graph = entry["connections"]

            max_attempts = 3
            attempts = 0
            success = False
            error_message = None

            while attempts < max_attempts and not success:
                try:
                    transitions, inference_time, total_tokens = generate_transitions_from_description(description, constraints, error_message)
                    print(f"Generated Transitions:\n{transitions}\n")  # Debug: Print extracted transitions
                    print(f"Inference Time: {inference_time} seconds")  # Print inference time
                    print(f"Total Token Count: {total_tokens}")  # Print total token count

                    is_legal = is_path_legal(transitions, nx.Graph(ground_truth_graph))
                    is_optimal = is_path_optimal(transitions, ground_truth)

                    results["total"] += 1
                    room_results[num_rooms]["total"] += 1
                    perturbed_results[perturbed]["total"] += 1

                    if is_legal:
                        results["legal_path"] += 1
                        room_results[num_rooms]["legal_path"] += 1
                        perturbed_results[perturbed]["legal_path"] += 1

                    if is_optimal:
                        results["optimal_path"] += 1
                        room_results[num_rooms]["optimal_path"] += 1
                        perturbed_results[perturbed]["optimal_path"] += 1

                    success = is_legal
                    if success:
                        results["success"] += 1
                        room_results[num_rooms]["success"] += 1
                        perturbed_results[perturbed]["success"] += 1
                        print(f"Success: Generated path is legal for num_rooms={num_rooms}, perturbed={perturbed}, entry {idx + 1}/{num_entries}")
                    else:
                        results["failure"] += 1
                        room_results[num_rooms]["failure"] += 1
                        perturbed_results[perturbed]["failure"] += 1
                        print(f"Failure: Generated path is not legal for num_rooms={num_rooms}, perturbed={perturbed}, entry {idx + 1}/{num_entries}")

                    results["details"].append({
                        "description": description,
                        "constraints": constraints,
                        "ground_truth_path": ground_truth,
                        "generated_path": transitions,
                        "ground_truth_graph": ground_truth_graph,
                        "generated_graph": ground_truth_graph,  # Assuming generated graph is the same as the ground truth graph
                        "success": success,
                        "legal_path": is_legal,
                        "optimal_path": is_optimal,
                        "attempts": attempts + 1,
                        "inference_time": inference_time,  # Log inference time
                        "token_count": total_tokens  # Log total token count
                    })

                except Exception as e:
                    print(e)
                    error_message = str(e)
                    attempts += 1
                    print("Failed, attempting again. Attempt number " + str(attempts))
                    if attempts >= max_attempts:
                        results["total"] += 1
                        results["failure"] += 1
                        room_results[num_rooms]["total"] += 1
                        room_results[num_rooms]["failure"] += 1
                        perturbed_results[perturbed]["total"] += 1
                        perturbed_results[perturbed]["failure"] += 1
                        print(f"Error: Maximum attempts reached for num_rooms={num_rooms}, perturbed={perturbed}, entry {idx + 1}/{num_entries} with error: {error_message}")
                        results["details"].append({
                            "description": description,
                            "constraints": constraints,
                            "ground_truth_path": ground_truth,
                            "generated_path": None,
                            "ground_truth_graph": ground_truth_graph,
                            "generated_graph": None,
                            "success": False,
                            "legal_path": False,
                            "optimal_path": False,
                            "error": error_message,
                            "attempts": attempts,
                            "inference_time": None,  # No inference time due to error
                            "token_count": total_tokens  # Log total token count even if there was an error
                        })

                # Save the results after each entry
                with open("navigation_results_newest_cot_no_code_distance.json", "w") as f:
                    json.dump(results, f, indent=4)
                with open("room_results_newest_cot_no_code_distance.json", "w") as f:
                    json.dump(room_results, f, indent=4)
                with open("perturbed_results_newest_cot_no_code_distance.json", "w") as f:
                    json.dump(perturbed_results, f, indent=4)

    return results, room_results, perturbed_results

if __name__ == "__main__":
    with open("test_dataset_distance.json", "r") as f:
        test_dataset = json.load(f)



  
    results, room_results, perturbed_results = test_llm_navigation(test_dataset, num_entries=100)

    print(f"Total: {results['total']}, Success: {results['success']}, Failure: {results['failure']}")
    print(f"Legal Paths: {results['legal_path']}, Optimal Paths: {results['optimal_path']}")

    with open("navigation_results_newest_cot_no_code_distance.json", "w") as f:
        json.dump(results, f, indent=4)
    with open("room_results_newest_cot_no_code_distance.json", "w") as f:
        json.dump(room_results, f, indent=4)
    with open("perturbed_results_newest_cot_no_code_distance.json", "w") as f:
        json.dump(perturbed_results, f, indent=4)
