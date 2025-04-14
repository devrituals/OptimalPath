import os
import pickle
import networkx as nx
import numpy as np
import traceback
from math import radians, cos, sin, asin, sqrt

# Create a cache directory if it doesn't exist
os.makedirs('cache', exist_ok=True)

# Try to import osmnx, but provide graceful fallback
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    print("Warning: osmnx package is not available. Road-based routing will use fallback methods.")
    OSMNX_AVAILABLE = False

# Try to import scikit-learn, which is needed by osmnx
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn package is not available. Some road network features will be limited.")
    SKLEARN_AVAILABLE = False

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    try:
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    except Exception as e:
        print(f"Error calculating haversine distance: {e}")
        # Return a fallback distance to avoid breaking calculations
        return 1.0  # Default to 1km if calculation fails

def get_road_network(location, network_type):
    """
    Get a road network for a location, with caching for better performance.
    Enhanced with better street network download options.
    
    Args:
        location: Location name or coordinates
        network_type: Type of network ('drive', 'walk', 'bike', etc.)
        
    Returns:
        G: NetworkX graph representing the road network
    """
    # Check if osmnx is available
    if not OSMNX_AVAILABLE:
        print("Error: osmnx is required to get road networks. Please install it using 'pip install osmnx'.")
        return None
        
    # Check if scikit-learn is available
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is required for road network operations.")
        print("Please install it using: pip install scikit-learn")
        return None
        
    # Create a safe cache filename
    cache_file = f'cache/{location.replace(" ", "_").replace(",", "").replace("/", "_")}_{network_type}.pkl'
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        try:
            print(f"Loaded road network from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached network: {e}")
            # If loading fails, continue to fetch a new network
    
    # If not cached or loading failed, fetch the network
    try:
        print(f"Fetching road network for {location}...")
        
        # Use more detailed parameters for better road networks
        # Include additional args to download a more complete network
        G = ox.graph_from_place(
            location, 
            network_type=network_type,
            simplify=True,  # Simplify the network topology
            retain_all=False,  # Discard disconnected nodes
            truncate_by_edge=True,  # Don't truncate nodes, only edges
            clean_periphery=True,  # Remove peripheral nodes
        )
        
        # Add travel time for better route calculation
        # speeds in km/h
        speed_dict = {
            'residential': 30,
            'primary': 60,
            'secondary': 50,
            'tertiary': 40,
            'unclassified': 30,
            'trunk': 80,
            'trunk_link': 50,
            'motorway': 100,
            'motorway_link': 60,
            'living_street': 20
        }
        
        # Add travel time (in seconds) to edges based on length and speed
        for u, v, k, data in G.edges(keys=True, data=True):
            # Get the road type, default to 'residential' if not specified
            highway_type = data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]  # Use the first classification if there are multiple
                
            # Get the speed for this type of road, default to 30 km/h
            speed = speed_dict.get(highway_type, 30)
            
            # Calculate travel time in seconds (length is in meters, speed in km/h)
            # Travel time = distance / speed (converted to appropriate units)
            # Convert speed from km/h to m/s: speed * 1000 / 3600
            speed_m_s = speed * 1000 / 3600
            data['travel_time'] = data['length'] / speed_m_s if speed_m_s > 0 else data['length']
        
        # Cache the network for future use
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
            print(f"Cached road network to {cache_file}")
        except Exception as e:
            print(f"Error caching network: {e}")
        
        return G
    except ImportError as e:
        if "scikit-learn" in str(e):
            print("Error: scikit-learn is required for road network operations.")
            print("Please install it using: pip install scikit-learn")
            return None
    except Exception as e:
        print(f"Error fetching road network: {e}")
        traceback.print_exc()
        return None

def get_nearest_node(G, coords):
    """
    Get the nearest node in the graph to the given coordinates.
    Handles the case when scikit-learn is not available by using
    a simple distance calculation instead of OSMnx's built-in function.
    
    Args:
        G: NetworkX graph representing the road network
        coords: (lat, lon) tuple
        
    Returns:
        nearest_node: Node ID of the nearest node
    """
    if G is None:
        print("Error: Road graph is None")
        return None
        
    try:
        # Check if osmnx and scikit-learn are available for optimal solution
        if OSMNX_AVAILABLE and SKLEARN_AVAILABLE:
            # Try using OSMnx's nearest node function
            return ox.distance.nearest_nodes(G, coords[1], coords[0])
        else:
            print("Using fallback method to find nearest node (dependencies not available)")
    except ImportError as e:
        print(f"Import error when finding nearest node: {e}")
        print("Using fallback method to find nearest node")
    except Exception as e:
        print(f"Error using OSMnx to find nearest node: {e}")
        print("Using fallback method to find nearest node")
    
    # Fallback: Calculate distances manually
    try:
        min_dist = float('inf')
        nearest_node = None
        
        for node, data in G.nodes(data=True):
            if 'y' in data and 'x' in data:
                dist = haversine_distance(coords[0], coords[1], data['y'], data['x'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
        
        if nearest_node is not None:
            return nearest_node
        else:
            # Last resort fallback: return the first node
            return list(G.nodes())[0]
    except Exception as e:
        print(f"Error in fallback nearest node method: {e}")
        traceback.print_exc()
        # Return a random node as a last resort
        try:
            return list(G.nodes())[0]
        except:
            return None

def find_shortest_path_on_road(G, start_coords, end_coords, weight='length', route_preference=None, traffic_factor=1.0, vehicle_factor=1.0):
    """
    Find the shortest path between two coordinates on the road network.
    Enhanced with better routing algorithm selection and weight handling.
    
    Args:
        G: NetworkX graph representing the road network
        start_coords: (lat, lon) tuple for starting point
        end_coords: (lat, lon) tuple for ending point
        weight: Edge attribute to minimize ('length' or 'travel_time')
        route_preference: String indicating routing preference (e.g., "Fastest route", "Avoid highways")
        traffic_factor: Factor to adjust travel time based on traffic conditions (>1 for heavier traffic)
        vehicle_factor: Factor to adjust travel time based on vehicle type (<1 for slower vehicles)
        
    Returns:
        path: List of node IDs in the shortest path
        cost: Total cost of the path
        path_coords: List of (lat, lon) coordinates along the path including road geometries
    """
    if G is None:
        print("Error: Road graph is None")
        return None, 0, None
        
    try:
        # Temporarily modify the graph based on route preferences and factors
        if weight == 'travel_time':
            # Create a temporary copy of the graph with adjusted weights
            temp_G = G.copy()
            
            # Apply traffic and vehicle factors to travel times
            for u, v, k, data in temp_G.edges(keys=True, data=True):
                if 'travel_time' in data:
                    # Adjust travel time based on traffic and vehicle
                    data['adjusted_time'] = data['travel_time'] * traffic_factor / vehicle_factor
                    
                    # Apply route preference adjustments
                    if route_preference:
                        highway_type = data.get('highway', 'residential')
                        if isinstance(highway_type, list):
                            highway_type = highway_type[0]
                        
                        # Different preferences modify the weights
                        if route_preference == "Fastest route":
                            # No additional adjustments for fastest route
                            pass
                        elif route_preference == "Shortest distance":
                            # For shortest distance, we'll still use the travel time but give less weight to it
                            data['adjusted_time'] = data['travel_time'] * 0.5 + data['length'] / 10000
                        elif route_preference == "Prefer main roads":
                            # Reduce the weight for main roads
                            if highway_type in ['primary', 'secondary', 'trunk', 'motorway']:
                                data['adjusted_time'] *= 0.8
                            elif highway_type in ['residential', 'living_street', 'unclassified']:
                                data['adjusted_time'] *= 1.2
                        elif route_preference == "Avoid highways":
                            # Increase the weight for highways
                            if highway_type in ['motorway', 'trunk', 'primary']:
                                data['adjusted_time'] *= 2.0
                        elif route_preference == "Scenic route":
                            # Prefer roads near water, parks, etc. (would need additional data)
                            # This is a simplified approximation
                            if 'residential' in highway_type or 'tertiary' in highway_type:
                                data['adjusted_time'] *= 0.9
            
            # Use the adjusted time as the weight
            weight_to_use = 'adjusted_time'
            G_to_use = temp_G
        else:
            # For distance-based routing, use the original graph and weight
            weight_to_use = weight
            G_to_use = G
        
        # Get the nearest nodes to the coordinates
        orig_node = get_nearest_node(G_to_use, start_coords)
        dest_node = get_nearest_node(G_to_use, end_coords)
        
        if orig_node is None or dest_node is None:
            print("Error: Could not find nearest nodes")
            # Use direct line as fallback
            distance = haversine_distance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
            return [0, 1], distance, [start_coords, end_coords]
        
        # Find the shortest path
        try:
            # Choose the right algorithm based on the weight
            if 'time' in weight_to_use:
                # Use A* for time-based routing which is better for time-based routing
                # The heuristic function is the straight-line distance divided by the maximum speed
                # Maximum speed is assumed to be 120 km/h (33.3 m/s)
                def heuristic(u, v):
                    u_y, u_x = G_to_use.nodes[u]['y'], G_to_use.nodes[u]['x']
                    v_y, v_x = G_to_use.nodes[v]['y'], G_to_use.nodes[v]['x']
                    
                    # Haversine distance in km
                    dist_km = haversine_distance(u_y, u_x, v_y, v_x)
                    
                    # Convert to meters
                    dist_m = dist_km * 1000
                    
                    # Estimate time in seconds (assuming max speed of 120 km/h = 33.3 m/s)
                    # Adjust for vehicle type and traffic
                    max_speed_m_s = 33.3 * vehicle_factor / traffic_factor
                    estimated_time = dist_m / max_speed_m_s
                    
                    return estimated_time
                
                try:
                    # First try A* with heuristic
                    path = nx.astar_path(G_to_use, orig_node, dest_node, heuristic, weight=weight_to_use)
                    
                    # Calculate total cost
                    cost = 0
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        edge_data = G_to_use.get_edge_data(u, v)
                        key = list(edge_data.keys())[0]
                        edge = edge_data[key]
                        cost += edge.get(weight_to_use, 0)
                except:
                    # Fall back to Dijkstra if A* fails
                    path = nx.dijkstra_path(G_to_use, orig_node, dest_node, weight=weight_to_use)
                    cost = nx.dijkstra_path_length(G_to_use, orig_node, dest_node, weight=weight_to_use)
            else:
                # Default to Dijkstra for distance-based routing
                path = nx.dijkstra_path(G_to_use, orig_node, dest_node, weight=weight_to_use)
                cost = nx.dijkstra_path_length(G_to_use, orig_node, dest_node, weight=weight_to_use)
            
            # Extract the detailed geometry points along the path
            path_coords = []
            
            # Add start point
            path_coords.append(start_coords)
            
            # Process each edge in the path to extract its geometry
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                
                # Check if this edge exists in the graph (it should)
                if G_to_use.has_edge(u, v):
                    # Get edge data (might have multiple edges between nodes)
                    edge_data = G_to_use.get_edge_data(u, v)
                    
                    # Get the first edge key (usually 0)
                    key = list(edge_data.keys())[0]
                    edge = edge_data[key]
                    
                    # Check if there's geometry data for this edge
                    if 'geometry' in edge and edge['geometry'] is not None:
                        # Extract all points from the LineString geometry
                        # The coords are in (lon, lat) format, so we need to swap them
                        geom_coords = [(point[1], point[0]) for point in list(edge['geometry'].coords)]
                        
                        # Add all intermediate points except the last one (to avoid duplicates)
                        if i < len(path) - 2:
                            path_coords.extend(geom_coords[:-1])
                        else:
                            # For the last segment, include all points
                            path_coords.extend(geom_coords)
                    else:
                        # If no geometry, just use node coordinates
                        if i == len(path) - 2:  # Last segment
                            u_y, u_x = G_to_use.nodes[u]['y'], G_to_use.nodes[u]['x']
                            v_y, v_x = G_to_use.nodes[v]['y'], G_to_use.nodes[v]['x']
                            path_coords.extend([(u_y, u_x), (v_y, v_x)])
                        else:
                            u_y, u_x = G_to_use.nodes[u]['y'], G_to_use.nodes[u]['x']
                            path_coords.append((u_y, u_x))
                else:
                    # This shouldn't happen, but just in case
                    try:
                        u_y, u_x = G_to_use.nodes[u]['y'], G_to_use.nodes[u]['x']
                        v_y, v_x = G_to_use.nodes[v]['y'], G_to_use.nodes[v]['x']
                        path_coords.extend([(u_y, u_x), (v_y, v_x)])
                    except KeyError:
                        print(f"Error: Missing coordinate data for nodes {u} or {v}")
            
            # Add end point to ensure it's included
            path_coords.append(end_coords)
            
            # Remove duplicate consecutive points
            cleaned_path_coords = []
            for i, coord in enumerate(path_coords):
                if i == 0 or coord != path_coords[i-1]:
                    cleaned_path_coords.append(coord)
            
            # Convert cost to km if using length, or minutes if using travel_time
            if weight == 'length' or weight_to_use == 'length':
                cost = cost / 1000  # Convert from meters to kilometers
            elif 'time' in weight_to_use:
                cost = cost / 60  # Convert from seconds to minutes
            
            return path, cost, cleaned_path_coords
        
        except nx.NetworkXNoPath:
            print(f"No path exists between the given coordinates")
            # Use direct line as fallback
            distance = haversine_distance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
            return [0, 1], distance, [start_coords, end_coords]
        
    except ImportError as e:
        if "scikit-learn" in str(e):
            print("Error in find_shortest_path_on_road: scikit-learn must be installed to search an unprojected graph")
            # Fallback to straight-line distance
            distance = haversine_distance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
            return [0, 1], distance, [start_coords, end_coords]
        else:
            print(f"Import error: {e}")
            return None, 0, None
    except Exception as e:
        print(f"Error finding shortest path on road: {e}")
        traceback.print_exc()
        # Use direct line as fallback
        distance = haversine_distance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        return [0, 1], distance, [start_coords, end_coords]

def create_distance_matrix_on_road(G, coords_list, weight='length', traffic_factor=1.0, vehicle_factor=1.0):
    """
    Create a distance matrix for the given coordinates using road distances.
    
    Args:
        G: NetworkX graph representing the road network
        coords_list: List of (lat, lon) tuples
        weight: Edge attribute to minimize ('length' or 'travel_time')
        traffic_factor: Factor to adjust travel time based on traffic conditions (>1 for heavier traffic)
        vehicle_factor: Factor to adjust travel time based on vehicle type (<1 for slower vehicles)
        
    Returns:
        distance_matrix: 2D array of distances between coordinates
        paths_dict: Dictionary of (i,j) -> (path, coords) for path visualization
    """
    if G is None:
        print("Error: Road graph is None, using straight-line distances")
        # Return straight-line distance matrix as fallback
        return create_straight_line_distance_matrix(coords_list), {}
        
    n = len(coords_list)
    distance_matrix = np.zeros((n, n))
    paths_dict = {}
    
    try:
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        # Pass traffic and vehicle factors to the path finding function
                        path, cost, path_coords = find_shortest_path_on_road(
                            G, coords_list[i], coords_list[j], weight, 
                            traffic_factor=traffic_factor, 
                            vehicle_factor=vehicle_factor
                        )
                        
                        if path:
                            # Apply traffic and vehicle factors to the cost if using travel_time
                            if weight == 'travel_time':
                                # Adjust travel time based on traffic and vehicle factors
                                cost = cost * traffic_factor / vehicle_factor
                            elif weight == 'balanced':
                                # For balanced, we'll use a weighted average of distance and time
                                # First get the distance
                                _, distance_cost, _ = find_shortest_path_on_road(
                                    G, coords_list[i], coords_list[j], 'length'
                                )
                                # Then get the time (adjusted by factors)
                                _, time_cost, _ = find_shortest_path_on_road(
                                    G, coords_list[i], coords_list[j], 'travel_time'
                                )
                                time_cost = time_cost * traffic_factor / vehicle_factor
                                
                                # Weighted average (60% time, 40% distance)
                                cost = 0.6 * time_cost + 0.4 * distance_cost
                            
                            distance_matrix[i][j] = cost
                            paths_dict[(i, j)] = (path, path_coords)
                        else:
                            # If no path exists, use straight-line distance as fallback
                            straight_line_dist = haversine_distance(
                                coords_list[i][0], coords_list[i][1], 
                                coords_list[j][0], coords_list[j][1]
                            )
                            distance_matrix[i][j] = straight_line_dist
                            paths_dict[(i, j)] = ([i, j], [coords_list[i], coords_list[j]])
                    except Exception as e:
                        print(f"Error computing distance from {i} to {j}: {e}")
                        # Use straight-line distance as fallback
                        straight_line_dist = haversine_distance(
                            coords_list[i][0], coords_list[i][1], 
                            coords_list[j][0], coords_list[j][1]
                        )
                        distance_matrix[i][j] = straight_line_dist
                        paths_dict[(i, j)] = ([i, j], [coords_list[i], coords_list[j]])
        
        return distance_matrix, paths_dict
    
    except ImportError as e:
        if "scikit-learn" in str(e):
            print("Error in create_distance_matrix_on_road: scikit-learn must be installed")
            # Fallback to straight-line distances
            return create_straight_line_distance_matrix(coords_list), create_straight_line_paths_dict(coords_list)
        else:
            print(f"Import error: {e}")
            return np.ones((n, n)) * 999999, {}
    except Exception as e:
        print(f"Error creating distance matrix: {e}")
        traceback.print_exc()
        # Fallback to straight-line distances
        return create_straight_line_distance_matrix(coords_list), create_straight_line_paths_dict(coords_list)

def create_straight_line_distance_matrix(coords_list):
    """
    Create a distance matrix using straight-line (haversine) distances
    
    Args:
        coords_list: List of (lat, lon) tuples
        
    Returns:
        distance_matrix: 2D array of distances between coordinates
    """
    n = len(coords_list)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = haversine_distance(
                    coords_list[i][0], coords_list[i][1], 
                    coords_list[j][0], coords_list[j][1]
                )
    
    return distance_matrix

def create_straight_line_paths_dict(coords_list):
    """
    Create a paths dictionary using straight lines between coordinates
    
    Args:
        coords_list: List of (lat, lon) tuples
        
    Returns:
        paths_dict: Dictionary of (i,j) -> (path, coords) for path visualization
    """
    n = len(coords_list)
    paths_dict = {}
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Create a simple path with just start and end points
                paths_dict[(i, j)] = ([i, j], [coords_list[i], coords_list[j]])
    
    return paths_dict

def convert_marrakech_graph_to_coordinates(G):
    """
    Convert the Marrakech graph to a dictionary of location -> coordinates
    
    Args:
        G: NetworkX graph representing Marrakech
        
    Returns:
        coords_dict: Dictionary mapping location names to (lat, lon) tuples
    """
    coords_dict = {}
    pos = nx.get_node_attributes(G, 'pos')
    
    for node, coords in pos.items():
        coords_dict[node] = coords
    
    return coords_dict