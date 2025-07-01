import random
import json
import networkx as nx

def join_with_and(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]
    
def generate_path_with_length(graph, rooms, path_length):
    while True:
        start_node = random.choice(rooms)
        end_node = random.choice([room for room in rooms if room != start_node])
        if nx.has_path(graph, start_node, end_node):
            path = nx.shortest_path(graph, start_node, end_node)
            if len(path) - 1 == path_length:
                return path, start_node, end_node
        else:
            # Create a new path with the required length
            path = [start_node]
            current_node = start_node
            for _ in range(path_length):
                next_node = random.choice([room for room in rooms if room not in path])
                graph.add_edge(current_node, next_node, weight=random.randint(1, 10))
                path.append(next_node)
                current_node = next_node
            return path, start_node, path[-1]

def generate_graph_with_constraints(rooms, min_path_length, max_path_length):
    graph = nx.Graph()
    for room in rooms:
        graph.add_node(room)
    
    path_length = random.randint(min_path_length, max_path_length)
    path, start_node, end_node = generate_path_with_length(graph, rooms, path_length)

    # Add additional connections without shortening the path
    for room in rooms:
        potential_connections = [r for r in rooms if r != room and not graph.has_edge(room, r)]
        if potential_connections:
            additional_connections = random.sample(potential_connections, random.randint(0, len(potential_connections)))
            for connected_room in additional_connections:
                graph.add_edge(room, connected_room, weight=random.randint(1, 10))
                if nx.shortest_path_length(graph, start_node, end_node) < path_length:
                    graph.remove_edge(room, connected_room)

    return graph, path, start_node, end_node

def generate_random_permutation(rooms, with_constraints=True, max_relax_attempts=3):
    min_path_length = (len(rooms) + 2) // 3
    max_path_length = (len(rooms) + 1) // 2

    graph, path, start_node, end_node = generate_graph_with_constraints(rooms, min_path_length, max_path_length)
    
    forbidden_nodes = []
    forbidden_edges = []
    if with_constraints:
        path_set = set(path)
        forbidden_nodes = random.sample(
            [room for room in rooms if room != start_node and room != end_node and room not in path_set],
            random.randint(1, min(3, len(rooms) - 2))
        )
        potential_edges = [
            (room1, room2) for room1 in rooms for room2 in rooms
            if room1 != room2 and room1 not in path_set and room2 not in path_set
        ]
        if potential_edges:
            forbidden_edges = random.sample(
                potential_edges,
                random.randint(1, min(5, len(potential_edges)))
            )
        # Remove forbidden nodes and edges from the graph
        graph.remove_nodes_from(forbidden_nodes)
        graph.remove_edges_from(forbidden_edges)

    # Ensure the ground truth path respects constraints
    for attempt in range(max_relax_attempts * 10):
        try:
            ground_truth_path = nx.shortest_path(graph, start_node, end_node)
            if not any(node in forbidden_nodes for node in ground_truth_path) and not any(
                (ground_truth_path[i], ground_truth_path[i + 1]) in forbidden_edges or 
                (ground_truth_path[i + 1], ground_truth_path[i]) in forbidden_edges 
                for i in range(len(ground_truth_path) - 1)
            ):
                break
        except nx.NetworkXNoPath:
            pass
    else:
        raise ValueError("Failed to find a valid ground truth path that respects constraints.")

    # Build ground truth connections
    ground_truth_connections = {node: {neighbor: graph.edges[node, neighbor]['weight'] for neighbor in neighbors} for node, neighbors in graph.adjacency()}

    # Initialize description
    description = "I have a house with the following rooms: " + ", ".join(rooms) + ".\n"

    # Keep track of mentioned pairs to avoid repetition
    mentioned_pairs = set()

    # Iterate through ground truth connections to build description
    for room, connected_rooms in ground_truth_connections.items():
        new_connections = [(connected_room, weight) for connected_room, weight in connected_rooms.items() if (room, connected_room) not in mentioned_pairs and (connected_room, room) not in mentioned_pairs]
        if new_connections:
            connection_descriptions = [f"{connected_room} with a distance of {weight}" for connected_room, weight in new_connections]
            description += f"{room} is connected to " + join_with_and(connection_descriptions) + ".\n"
            mentioned_pairs.update((room, connected_room) for connected_room, weight in new_connections)
            mentioned_pairs.update((connected_room, room) for connected_room, weight in new_connections)

    # Build constraints
    constraints = f"Start in {start_node} and go to {end_node}"
    if forbidden_nodes or forbidden_edges:
        constraints += " without passing through " + join_with_and(forbidden_nodes) + "."
        if forbidden_edges:
            constraints += " Avoid moving from " + join_with_and([f"{edge[0]} directly into {edge[1]}" for edge in forbidden_edges]) + "."

    # Return the required values
    return description.strip(), constraints.strip(), {}, ground_truth_connections, path, len(path) - 1, ground_truth_path

def create_test_dataset(file_name, num_entries_per_section=100):
    dataset = []
    room_counts = [5, 10, 15, 20, 25, 30]

    for num_rooms in room_counts:
        generated_count = 0
        
        while generated_count < num_entries_per_section:
            rooms = [f"Room{i}" for i in range(1, num_rooms + 1)]
            
            try:
                # With constraints
                description, constraints, _, ground_truth_connections, path, path_length, ground_truth_path = generate_random_permutation(rooms, with_constraints=True)
                
                # Reconstruct the graph from connections
                graph = nx.Graph()
                for room, neighbors in ground_truth_connections.items():
                    for neighbor, weight in neighbors.items():
                        graph.add_edge(room, neighbor, weight=weight)

                # Extract start and end nodes from constraints
                start_node = constraints.split("Start in ")[1].split(" and go to ")[0]
                end_node = constraints.split(" and go to ")[1].split(" without")[0].strip()

                shortest_path = nx.shortest_path(graph, start_node, end_node, weight='weight')

                entry = {
                    "num_rooms": num_rooms,
                    "perturbed": True,
                    "description": description,
                    "constraints": constraints,
                    "connections": ground_truth_connections,
                    "ground_truth": shortest_path,
                }
                dataset.append(entry)
                generated_count += 1
                
                # Without constraints
                description, constraints, _, ground_truth_connections, path, path_length, ground_truth_path = generate_random_permutation(rooms, with_constraints=False)
                graph = nx.Graph()
                for room, neighbors in ground_truth_connections.items():
                    for neighbor, weight in neighbors.items():
                        graph.add_edge(room, neighbor, weight=weight)

                # Extract start and end nodes from constraints
                start_node = constraints.split("Start in ")[1].split(" and go to ")[0]
                end_node = constraints.split(" and go to ")[1].split(" without")[0].strip()

                shortest_path = nx.shortest_path(graph, start_node, end_node, weight='weight')

                entry = {
                    "num_rooms": num_rooms,
                    "perturbed": False,
                    "description": description,
                    "constraints": constraints,
                    "connections": ground_truth_connections,
                    "ground_truth": shortest_path,
                }
                dataset.append(entry)
                generated_count += 1
            except ValueError as e:
                print(f"Skipping generation due to error: {e}")
                continue

    with open(file_name, 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    create_test_dataset("test_dataset_paths_weighted.json")
