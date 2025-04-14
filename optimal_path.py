import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import traceback
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
        if start not in graph:
            print(f"Warning: Start location '{start}' not in graph")
            return None, float('inf')
        
        if end not in graph:
            print(f"Warning: End location '{end}' not in graph")
            return None, float('inf')
            
        path = nx.shortest_path(graph, start, end, weight=cost_type)
        cost = nx.shortest_path_length(graph, start, end, weight=cost_type)
        return path, cost
    except nx.NetworkXNoPath:
        print(f"No path exists between {start} and {end}")
        return None, float('inf')
    except Exception as e:
        print(f"Error finding shortest path: {e}")
        traceback.print_exc()
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
    try:
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
                        # Use integer values for OR-Tools compatibility, but preserve precision
                        # by multiplying small values (like distances in km)
                        if cost_type == 'distance' and cost < 100:  
                            # Convert km to meters for better precision with integers
                            distance_matrix[i][j] = int(cost * 1000)
                        else:
                            distance_matrix[i][j] = int(cost)
        
        return distance_matrix.astype(int)
    except Exception as e:
        print(f"Error creating distance matrix: {e}")
        traceback.print_exc()
        # Return a matrix of large values if error occurs
        return np.ones((len(locations), len(locations)), dtype=int) * 999999


def solve_tsp(distance_matrix, start_index=0, delivery_person_location=None, locations=None, time_limit_seconds=30, algorithm="auto", round_trip=False):
    """
    Solve the Traveling Salesman Problem using OR-Tools with enhanced optimization.
    
    Args:
        distance_matrix: 2D array of distances between locations
        start_index: Index of the starting location
        delivery_person_location: Location of the delivery person
        locations: List of all locations
        time_limit_seconds: Time limit for the solver in seconds
        algorithm: Algorithm to use for solving TSP ("auto", "nearest_neighbor", "greedy", "two_opt", "simulated_annealing", "genetic")
        round_trip: Whether to return to the starting point (True) or not (False)
        
    Returns:
        route: List of indices representing the optimal route
    """
    try:
        n = len(distance_matrix)
        
        if n <= 1:
            print("Warning: Need at least 2 locations to find a route")
            return [0] if n == 1 else []
        
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
        
        # Create a routing model
        if round_trip:
            # For round trips, we end at the same location we started
            manager = pywrapcp.RoutingIndexManager(n, 1, start_index)
        else:
            # For open routes (no return to start), we create a dummy end point
            # We use the same index as start but mark it as a different node in the manager
            manager = pywrapcp.RoutingIndexManager(n, 1, [start_index], [start_index])
        
        routing = pywrapcp.RoutingModel(manager)
        
        # Create a callback to calculate distance between points
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Scale for better precision
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters - try multiple strategies
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Add more time for complex problems
        search_parameters.time_limit.seconds = time_limit_seconds
        
        # Map the algorithm parameter to OR-Tools strategies
        algorithm_strategies = {
            "nearest_neighbor": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            "greedy": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            "two_opt": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
            "simulated_annealing": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
            "genetic": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        }
        
        # Choose strategies based on algorithm parameter
        if algorithm in algorithm_strategies:
            # Use the specified algorithm
            solution_strategies = [algorithm_strategies[algorithm]]
        elif algorithm == "auto":
            # Try multiple solution strategies in order of increasing complexity
            solution_strategies = [
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
                routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
                routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
                routing_enums_pb2.FirstSolutionStrategy.SWEEP
            ]
        else:
            # Default to a balanced approach
            solution_strategies = [
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
                routing_enums_pb2.FirstSolutionStrategy.SAVINGS
            ]
        
        # Set local search metaheuristic based on algorithm
        if algorithm == "simulated_annealing":
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
        elif algorithm == "genetic":
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)  # More intensive search
            search_parameters.guided_local_search_lambda_coefficient = 0.8  # More exploration
        else:
            # Use guided local search as default
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.guided_local_search_lambda_coefficient = 0.5
        
        # Increase efforts for more optimal results
        search_parameters.log_search = True
        
        # If not a round trip, set the end node constraint
        if not round_trip:
            # Allow ending at any node (no need to return to start)
            for i in range(n):
                routing.SetAllowedVehicleEndIndices(0, [manager.NodeToIndex(i)])
        
        # Try different solutions and use the best one
        best_solution = None
        best_objective = float('inf')
        
        for strategy in solution_strategies:
            search_parameters.first_solution_strategy = strategy
            current_solution = routing.SolveWithParameters(search_parameters)
            
            if current_solution:
                if routing.GetObjectiveValue(current_solution) < best_objective:
                    best_solution = current_solution
                    best_objective = routing.GetObjectiveValue(current_solution)
        
        solution = best_solution
        
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
                
            # For non-round trips, avoid adding the same node twice at the end
            if not round_trip and len(route) > 0 and manager.IndexToNode(index) == route[0]:
                pass  # Don't add the starting point again
            else:
                route.append(manager.IndexToNode(index))  # Add the final node
                
            return route
        else:
            print("Failed to find a solution to the TSP.")
            # If no solution is found, just return a sequential route as fallback
            return list(range(n))
    except Exception as e:
        print(f"Error solving TSP: {e}")
        traceback.print_exc()
        # Return a simple sequential route as fallback
        return list(range(min(n, 1)))


# Step 4: Visualize the route
def plot_graph(graph, path=None, title="Marrakech Map"):
    """
    Visualize the graph and highlight the path if provided
    
    Args:
        graph: NetworkX graph
        path: List of locations in the path to highlight
        title: Title of the plot
    """
    try:
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
    except Exception as e:
        print(f"Error plotting graph: {e}")
        traceback.print_exc()

# Main function to test the implementation
def main():
    try:
        # Create the graph
        G = create_marrakech_sample_graph()
        
        # Test Dijkstra's algorithm
        start_loc = 'Jemaa el-Fnaa'
        end_loc = 'Majorelle Garden'
        path, cost = find_shortest_path(G, start_loc, end_loc, 'distance')
        print(f"Shortest path from {start_loc} to {end_loc} by distance:")
        if path:
            print(f"Path: {path}")
            print(f"Total distance: {cost} km\n")
            
            # Visualize the shortest path
            plot_graph(G, path, f"Shortest Path from {start_loc} to {end_loc}")
        else:
            print("No path found.\n")
        
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
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()