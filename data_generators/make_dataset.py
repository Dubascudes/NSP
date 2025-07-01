import random
import json
import networkx as nx

def generate_random_connections(rooms):
    connections = {}
    for room in rooms:
        connected_rooms = random.sample([r for r in rooms if r != room], random.randint(1, int(len(rooms) * .15)))
        if room not in connections:
            connections[room] = {}
        for connected_room in connected_rooms:
            distance = round(random.uniform(1.0, 15.0), 2)
            connections[room][connected_room] = distance
            if connected_room not in connections:
                connections[connected_room] = {}
            connections[connected_room][room] = distance  # Ensure the graph is undirected
    return connections

def join_with_and(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]

def generate_random_permutation(rooms, with_constraints=True, max_relax_attempts=3):
    connections = generate_random_connections(rooms)
    
    graph = nx.Graph()
    for room, connected_rooms in connections.items():
        for connected_room, distance in connected_rooms.items():
            graph.add_edge(room, connected_room, weight=distance)
    
    # Ensure the graph is connected
    components = list(nx.connected_components(graph))
    while len(components) > 1:
        node_from_comp1 = random.choice(list(components[0]))
        node_from_comp2 = random.choice(list(components[1]))
        distance = round(random.uniform(1.0, 15.0), 2)
        graph.add_edge(node_from_comp1, node_from_comp2, weight=distance)
        components = list(nx.connected_components(graph))  # update components

    max_attempts = 100
    attempts = 0
    forbidden_nodes = []
    forbidden_edges = []

    while attempts < max_attempts:
        start_node = random.choice(rooms)
        end_node = random.choice([room for room in rooms if room != start_node])
        try:
            path = nx.shortest_path(graph, start_node, end_node, weight='weight')
            path_length = nx.shortest_path_length(graph, start_node, end_node, weight='weight')
            
            if with_constraints:
                path_set = set(path)
                forbidden_nodes = random.sample([room for room in rooms if room != start_node and room != end_node and room not in path_set], random.randint(1, min(3, len(rooms)-2)))
                forbidden_edges = random.sample([(room1, room2) for room1 in rooms for room2 in rooms if room1 != room2 and room1 not in path_set and room2 not in path_set], random.randint(1, min(5, len(rooms)*(len(rooms)-1)//2)))

            if not any(node in forbidden_nodes for node in path) and not any((path[i], path[i+1]) in forbidden_edges or (path[i+1], path[i]) in forbidden_edges for i in range(len(path) - 1)):
                break
        except nx.NetworkXNoPath:
            pass

        attempts += 1
        # Relax constraints if no valid path is found
        if attempts % (max_attempts // max_relax_attempts) == 0:
            if forbidden_nodes:
                forbidden_nodes.pop()
            elif forbidden_edges:
                forbidden_edges.pop()
            else:
                break
    else:
        raise ValueError("Failed to find a valid path in a connected graph with given constraints. This should not happen.")

    # Introduce an obstacle along the initial path to create a ground truth graph different from the initial graph
    ground_truth_graph = graph.copy()
    # edge_to_remove = random.choice(list(zip(path[:-1], path[1:])))
    # ground_truth_graph.remove_edge(*edge_to_remove)

    # Ensure the ground truth graph is still connected and find a new path that respects constraints
    components = list(nx.connected_components(ground_truth_graph))
    while len(components) > 1:
        node_from_comp1 = random.choice(list(components[0]))
        node_from_comp2 = random.choice(list(components[1]))
        distance = round(random.uniform(1.0, 15.0), 2)
        ground_truth_graph.add_edge(node_from_comp1, node_from_comp2, weight=distance)
        components = list(nx.connected_components(ground_truth_graph))  # update components

    # Ensure new ground truth path respects constraints
    for attempt in range(max_attempts):
        try:
            ground_truth_path = nx.shortest_path(ground_truth_graph, start_node, end_node, weight='weight')
            ground_truth_path_length = nx.shortest_path_length(ground_truth_graph, start_node, end_node, weight='weight')
            if not any(node in forbidden_nodes for node in ground_truth_path) and not any((ground_truth_path[i], ground_truth_path[i+1]) in forbidden_edges or (ground_truth_path[i+1], ground_truth_path[i]) in forbidden_edges for i in range(len(ground_truth_path) - 1)):
                break
        except nx.NetworkXNoPath:
            pass
    else:
        raise ValueError("Failed to find a valid ground truth path that respects constraints.")

    # Build ground truth connections
    ground_truth_connections = {node: {neighbor: data['weight'] for neighbor, data in neighbors.items()} for node, neighbors in ground_truth_graph.adjacency()}

    # Initialize description and set for described edges
    description = "I have a house with the following rooms: " + ", ".join(rooms) + ".\n"
    described_edges = set()

    # Iterate through connections to build description
    for room, connected_rooms in connections.items():
        for connected_room, distance in connected_rooms.items():
            # Ensure edge is described only once
            edge = frozenset([room, connected_room])  # Use frozenset for unordered pair
            if edge not in described_edges:
                description += f"{room} is connected to {connected_room} with a distance of {distance}.\n"
                described_edges.add(edge)

    # Build constraints
    constraints = f"Start in {start_node} and go to {end_node}"
    if forbidden_nodes or forbidden_edges:
        constraints += " without passing through " + join_with_and(forbidden_nodes) + "."
        if forbidden_edges:
            constraints += " Avoid moving from " + join_with_and([f"{edge[0]} directly into {edge[1]}" for edge in forbidden_edges]) + "."

    # Return the required values
    return description.strip(), constraints.strip(), connections, ground_truth_connections, path, path_length, ground_truth_path, ground_truth_path_length

def create_test_dataset(file_name, num_entries_per_section=100):
    dataset = []
    room_counts = [15, 25]

    for num_rooms in room_counts:
        generated_count = 0
        
        while generated_count < num_entries_per_section:
            rooms = [f"Room{i}" for i in range(1, num_rooms + 1)]
            
            try:
                # With constraints
                description, constraints, connections, ground_truth_connections, path, path_length, ground_truth_path, ground_truth_path_length = generate_random_permutation(rooms, with_constraints=True)
                entry = {
                    "num_rooms": num_rooms,
                    "perturbed": True,
                    "description": description,
                    "constraints": constraints,
                    # "initial_connections": connections,
                    "connections": ground_truth_connections,
                    "ground_truth": ground_truth_path,
                    # "initial_path": path,
                    # "initial_path_length": path_length,
                    "ground_truth_path_length": ground_truth_path_length
                }
                dataset.append(entry)
                generated_count += 1
                
                # Without constraints
                description, constraints, connections, ground_truth_connections, path, path_length, ground_truth_path, ground_truth_path_length = generate_random_permutation(rooms, with_constraints=False)
                entry = {
                    "num_rooms": num_rooms,
                    "perturbed": False,
                    "description": description,
                    "constraints": constraints,
                    # "initial_connections": connections,
                    "connections": ground_truth_connections,
                    "ground_truth": ground_truth_path,
                    # "initial_path": path,
                    # "initial_path_length": path_length,
                    "ground_truth_path_length": ground_truth_path_length
                }
                dataset.append(entry)
                generated_count += 1
            except ValueError as e:
                print(f"Skipping generation due to error: {e}")
                continue

    with open(file_name, 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    create_test_dataset("test_dataset_dist15_25.json")
