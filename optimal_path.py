import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Step 1: Create a simple graph representing key locations in Marrakech
def create_marrakech_sample_graph():
    """
    Create a simple graph with some key locations in Marrakech
    Nodes represent locations, edges represent roads with distance and time attributes
    """
    G = nx.Graph()
    
    # Add some key locations in Marrakech as nodes
    # These are simplified for demonstration purposes
    locations = {
        'Jemaa el-Fnaa': (31.6258, -7.9891),
        'Koutoubia Mosque': (31.6248, -7.9933),
        'Bahia Palace': (31.6216, -7.9828),
        'Majorelle Garden': (31.6417, -7.9988),
        'Menara Gardens': (31.6147, -8.0162),
        'El Badi Palace': (31.6184, -7.9832),
        'Saadian Tombs': (31.6178, -7.9850),
        'Medina': (31.6295, -7.9811),
        'Gueliz': (31.6370, -8.0122),
        'Marrakech Train Station': (31.6484, -8.0142)
    }
    
    # Add nodes with position attributes for visualization
    for loc, coords in locations.items():
        G.add_node(loc, pos=coords)

    
    # Add edges with distance (km) and time (minutes) attributes
    # These values are approximated for demonstration
    edges = [
        ('Jemaa el-Fnaa', 'Koutoubia Mosque', {'distance': 0.5, 'time': 7}),
        ('Jemaa el-Fnaa', 'Bahia Palace', {'distance': 1.2, 'time': 15}),
        ('Jemaa el-Fnaa', 'Medina', {'distance': 0.8, 'time': 12}),
        ('Koutoubia Mosque', 'Menara Gardens', {'distance': 2.5, 'time': 20}),
        ('Koutoubia Mosque', 'Gueliz', {'distance': 2.2, 'time': 18}),
        ('Bahia Palace', 'El Badi Palace', {'distance': 0.4, 'time': 5}),
        ('Bahia Palace', 'Saadian Tombs', {'distance': 0.5, 'time': 7}),
        ('Majorelle Garden', 'Gueliz', {'distance': 1.0, 'time': 12}),
        ('Majorelle Garden', 'Marrakech Train Station', {'distance': 1.5, 'time': 15}),
        ('Menara Gardens', 'Gueliz', {'distance': 2.0, 'time': 18}),
        ('El Badi Palace', 'Saadian Tombs', {'distance': 0.3, 'time': 4}),
        ('Medina', 'Bahia Palace', {'distance': 1.0, 'time': 14}),
        ('Gueliz', 'Marrakech Train Station', {'distance': 1.2, 'time': 10}),
        ('Medina', 'Koutoubia Mosque', {'distance': 1.1, 'time': 15}),
        ('Jemaa el-Fnaa', 'El Badi Palace', {'distance': 1.0, 'time': 13}),
    ]
    
    G.add_edges_from(edges)
    return G

# Step 2: Implement Dijkstra's algorithm for shortest path
def find_shortest_path(graph, start, end, cost_type='distance'):
    """
    Find the shortest path between two locations using Dijkstra's algorithm
    
    Args:
        graph: NetworkX graph
        start: Starting location
        end: Ending location
        cost_type: 'distance' or 'time'
        
    Returns:
        path: List of locations in the shortest path
        cost: Total cost (distance or time) of the path
    """
    try:
        path = nx.shortest_path(graph, start, end, weight=cost_type)
        cost = nx.shortest_path_length(graph, start, end, weight=cost_type)
        return path, cost
    except nx.NetworkXNoPath:
        return None, float('inf')

# Step 3: Implement TSP for multi-stop optimization
def create_distance_matrix(graph, locations, cost_type='distance'):
    """
    Create a distance matrix for the given locations
    
    Args:
        graph: NetworkX graph
        locations: List of locations
        cost_type: 'distance' or 'time'
        
    Returns:
        distance_matrix: 2D array of distances between locations
    """
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                path, cost = find_shortest_path(graph, locations[i], locations[j], cost_type)
                if path is None:
                    # If no path exists, set a very large value
                    distance_matrix[i][j] = 999999
                else:
                    distance_matrix[i][j] = cost
    
    return distance_matrix.astype(int)

def solve_tsp(distance_matrix, start_index=0, delivery_person_location=None, locations=None):
    """
    Solve the Traveling Salesman Problem using OR-Tools
    
    Args:
        distance_matrix: 2D array of distances between locations
        start_index: Index of the starting location
        delivery_person_location: Location of the delivery person
        locations: List of all locations
        
    Returns:
        route: List of indices representing the optimal route
    """
    n = len(distance_matrix)
    
    # If delivery person location is provided, find the nearest location to start from
    if delivery_person_location and locations:
        # Create a graph to find distances from delivery person to all locations
        G = create_marrakech_sample_graph()
        
        # Find the nearest location to the delivery person
        min_distance = float('inf')
        nearest_index = 0
        
        for i, loc in enumerate(locations):
            path, cost = find_shortest_path(G, delivery_person_location, loc, 'distance')
            if path and cost < min_distance:
                min_distance = cost
                nearest_index = i
        
        # Set the nearest location as the starting point
        start_index = nearest_index
    
    manager = pywrapcp.RoutingIndexManager(n, 1, start_index)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Add the final node
        return route
    return None

# Step 4: Visualize the route
def plot_graph(graph, path=None, title="Marrakech Map"):
    """
    Visualize the graph and highlight the path if provided
    
    Args:
        graph: NetworkX graph
        path: List of locations in the path to highlight
        title: Title of the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Get positions from node attributes
    pos = nx.get_node_attributes(graph, 'pos')
    
    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')
    
    # Highlight the path if provided
    if path and len(path) > 1:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_size=500, node_color='red')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=3, edge_color='red')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('marrakech_route.png')
    plt.close()

# Main function to test the implementation
def main():
    # Create the graph
    G = create_marrakech_sample_graph()
    
    # Test Dijkstra's algorithm
    start_loc = 'Jemaa el-Fnaa'
    end_loc = 'Majorelle Garden'
    path, cost = find_shortest_path(G, start_loc, end_loc, 'distance')
    print(f"Shortest path from {start_loc} to {end_loc} by distance:")
    print(f"Path: {path}")
    print(f"Total distance: {cost} km\n")
    
    # Visualize the shortest path
    plot_graph(G, path, f"Shortest Path from {start_loc} to {end_loc}")
    
    # Test TSP for multiple stops
    locations = ['Jemaa el-Fnaa', 'Bahia Palace', 'Majorelle Garden', 'Menara Gardens', 'Marrakech Train Station']
    print(f"Finding optimal route through {len(locations)} locations:")
    for loc in locations:
        print(f"- {loc}")
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(G, locations, 'distance')
    print("\nDistance Matrix:")
    print(distance_matrix)
    
    # Solve TSP
    route_indices = solve_tsp(distance_matrix)
    if route_indices:
        route = [locations[i] for i in route_indices]
        print("\nOptimal Route:")
        for i, loc in enumerate(route):
            print(f"{i+1}. {loc}")
        
        # Visualize the TSP route
        plot_graph(G, route, "Optimal Multi-Stop Route in Marrakech")
    else:
        print("\nNo valid route found.")

if __name__ == "__main__":
    main()
