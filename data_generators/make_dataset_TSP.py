import random
import json
import networkx as nx
from networkx.algorithms.approximation import greedy_tsp

def generate_tsp_cycle(rooms):
    """ Generate a random TSP cycle """
    nodes = list(range(len(rooms)))
    random.shuffle(nodes)
    cycle = nodes + [nodes[0]]  # Create a cycle
    return cycle

def generate_graph_from_tsp_cycle(rooms, cycle, weighted=True, directed=False):
    """ Generate a graph based on the TSP cycle """
    graph = nx.DiGraph() if directed else nx.Graph()
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        weight = round(random.uniform(1.0, 15.0), 2) if weighted else 1
        graph.add_edge(rooms[u], rooms[v], weight=weight)
    return graph

def add_additional_edges(graph, rooms, retention_ratio=0.3, weighted=True):
    """ Add additional edges to the graph based on the retention ratio """
    complete_graph = nx.complete_graph(len(rooms))
    for u, v in complete_graph.edges():
        complete_graph[u][v]['weight'] = round(random.uniform(1.0, 15.0), 2) if weighted else 1
    
    # Renaming nodes to room names for readability
    mapping = {i: f"Room{i+1}" for i in range(len(rooms))}
    complete_graph = nx.relabel_nodes(complete_graph, mapping)
    
    additional_edges = [edge for edge in complete_graph.edges if not graph.has_edge(*edge)]
    additional_edges = random.sample(additional_edges, int(len(additional_edges) * retention_ratio))
    
    for u, v in additional_edges:
        graph.add_edge(u, v, weight=complete_graph[u][v]['weight'])

def generate_tsp_problem(rooms, weighted=True, directed=False):
    # Generate a TSP cycle
    cycle = generate_tsp_cycle(rooms)

    # Create a graph based on the TSP cycle
    graph = generate_graph_from_tsp_cycle(rooms, cycle, weighted, directed)

    # Add additional edges to the graph
    add_additional_edges(graph, rooms, weighted=weighted)

    # Generate the distance graph description
    description = f"Find a path that visits all rooms exactly once, beginning and ending in {rooms[cycle[0]]}. The house has the following rooms: " + ", ".join(rooms) + ".\n"
    for u, v in graph.edges():
        if weighted:
            description += f"{u} is connected to {v} with a distance of {graph[u][v]['weight']}.\n"
        else:
            description += f"{u} is connected to {v}.\n"
    distance_graph = {node: {neighbor: data['weight'] for neighbor, data in neighbors.items()} for node, neighbors in graph.adjacency()}

    return {
        "num_rooms": len(rooms),
        "description": description,
        "cycle": [rooms[i] for i in cycle],
        "distance_graph": distance_graph,
        "graph_type": "directed" if directed else "weighted" if weighted else "unweighted"
    }

def create_tsp_dataset(file_name, num_entries=250):
    dataset = []
    room_counts = [5, 10, 15, 20, 25]  # Different sizes of rooms

    types_of_graphs = [('weighted', True, False)]

    entries_per_type = num_entries // len(room_counts) // len(types_of_graphs)

    for num_rooms in room_counts:
        for graph_type, weighted, directed in types_of_graphs:
            for _ in range(entries_per_type):
                rooms = [f"Room{i}" for i in range(1, num_rooms + 1)]
                tsp_data = generate_tsp_problem(rooms, weighted, directed)
                dataset.append(tsp_data)

    with open(file_name, 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    create_tsp_dataset("cleaned_dataset_weighted_tsp.json")
