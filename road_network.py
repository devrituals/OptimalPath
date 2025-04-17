import osmnx as ox
import networkx as nx
import numpy as np
import folium
import time
import os
import pickle
from pathlib import Path

# Ensure we have the right OSMnx configuration
try:
    # Configure OSMnx settings - use newer API if available
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 180  # Increase timeout for better reliability
    ox.settings.memory = 4000  # Increase memory limit for larger networks
    print("Using newer OSMnx API")
except AttributeError:
    # Fallback to older configuration method
    ox.config(use_cache=True, log_console=False, timeout=180)
    print("Using older OSMnx API")

def get_road_network(place_name="Marrakech, Morocco", network_type="drive", force_refresh=False):
    """
    Get a road network for a specific place using OSMnx with enhanced reliability.
    
    Args:
        place_name: Name of the place to get the road network for
        network_type: Type of network to get ('drive', 'bike', 'walk', etc.)
        force_refresh: Whether to force refreshing the cache
        
    Returns:
        G: NetworkX graph representing the road network
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create a cache filename based on the place name and network type
    safe_place_name = place_name.replace(' ', '_').replace(',', '').replace('/', '_')
    cache_filename = cache_dir / f"{safe_place_name}_{network_type}.pkl"
    
    # Check if the cache file exists and we're not forcing a refresh
    if cache_filename.exists() and not force_refresh:
        try:
            with open(cache_filename, 'rb') as f:
                G = pickle.load(f)
                print(f"Loaded road network from cache: {cache_filename}")
                
                # Verify the graph has edges
                if G.number_of_edges() > 0:
                    return G
                else:
                    print("Cached graph has no edges, getting a new graph")
        except Exception as e:
            print(f"Error loading cached road network: {e}")
    
    # Define box around Marrakech to ensure we get a complete network
    # This is a fallback in case place name lookup fails
    marrakech_bbox = (31.57, 31.68, -8.04, -7.93)  # (south, north, west, east)
    
    # Try multiple approaches to get the road network
    for attempt in range(3):
        try:
            print(f"Attempt {attempt+1} to get road network for {place_name}")
            
            # First, try the place name approach
            try:
                print("Trying to get network by place name...")
                try:
                    # For newer versions of OSMnx
                    G = ox.graph_from_place(place_name, network_type=network_type)
                except TypeError:
                    # For older versions of OSMnx
                    G = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
            except Exception as e:
                print(f"Error getting network by place name: {e}")
                print("Falling back to bounding box...")
                
                # Fallback to bounding box if place name fails
                try:
                    # For newer versions of OSMnx
                    G = ox.graph_from_bbox(
                        marrakech_bbox[0], marrakech_bbox[1], 
                        marrakech_bbox[2], marrakech_bbox[3], 
                        network_type=network_type
                    )
                except TypeError:
                    # For older versions of OSMnx
                    G = ox.graph_from_bbox(
                        marrakech_bbox[0], marrakech_bbox[1], 
                        marrakech_bbox[2], marrakech_bbox[3], 
                        network_type=network_type, simplify=True
                    )
            
            # If we got a graph with no nodes or edges, try again with different params
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                print(f"Got an empty graph (nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()})")
                raise ValueError("Empty graph returned")
            
            # Project the graph to UTM for proper distance calculations
            try:
                # For newer versions of OSMnx
                G = ox.project_graph(G)
            except TypeError:
                # For older versions of OSMnx
                G = ox.project_graph(G, to_crs=None)
            
            # Add edge speeds and travel times
            try:
                # For newer versions of OSMnx
                G = ox.add_edge_speeds(G)
                G = ox.add_edge_travel_times(G)
            except (TypeError, AttributeError):
                # For older versions of OSMnx
                G = ox.add_edge_speeds(G, hwy_speeds=None)
                G = ox.add_edge_travel_times(G, edge_speeds=None)
            
            # Verify the graph has the necessary attributes
            for _, _, data in G.edges(data=True):
                if 'length' not in data:
                    data['length'] = 100  # Default length in meters
                if 'travel_time' not in data:
                    data['travel_time'] = 60  # Default time in seconds
            
            # Save the network to cache
            try:
                with open(cache_filename, 'wb') as f:
                    pickle.dump(G, f)
                    print(f"Saved road network to cache: {cache_filename}")
            except Exception as e:
                print(f"Error saving road network to cache: {e}")
            
            print(f"Successfully created road network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)  # Wait before retry
    
    print("All attempts to get road network failed, creating fallback network")
    # If all attempts fail, create a simplified fallback network
    return create_fallback_network()

def create_fallback_network():
    """
    Create a simplified fallback network for Marrakech when OSMnx fails.
    """
    G = nx.MultiDiGraph()
    
    # Add some key locations in Marrakech as nodes
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
    
    # Add nodes with position attributes
    for name, (lat, lon) in locations.items():
        # Add node with both y/x and lat/lng attributes for compatibility
        G.add_node(name, y=lat, x=lon, lat=lat, lon=lon)
    
    # Connect all nodes to create a complete graph (every node connects to every other node)
    nodes = list(G.nodes())
    for i, node1 in enumerate(nodes):
        y1, x1 = G.nodes[node1]['y'], G.nodes[node1]['x']
        for node2 in nodes[i+1:]:
            y2, x2 = G.nodes[node2]['y'], G.nodes[node2]['x']
            
            # Calculate Euclidean distance (in meters) and travel time
            # 111,320 is approx. meters per degree of latitude
            # cos(lat) * 111,320 is approx. meters per degree of longitude at given latitude
            import math
            dx = (x2 - x1) * 111320 * math.cos(math.radians((y1 + y2) / 2))
            dy = (y2 - y1) * 111320
            distance = math.sqrt(dx*dx + dy*dy)  # meters
            
            # Assume average speed of 30 km/h = 8.33 m/s
            travel_time = distance / 8.33  # seconds
            
            # Add bidirectional edges
            G.add_edge(node1, node2, length=distance, travel_time=travel_time, highway='path')
            G.add_edge(node2, node1, length=distance, travel_time=travel_time, highway='path')
    
    print(f"Created fallback network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def get_nearest_node(G, point):
    """
    Get the nearest node in the graph to a point.
    
    Args:
        G: NetworkX graph
        point: (lat, lon) tuple
        
    Returns:
        node_id: ID of the nearest node
    """
    try:
        # Try using the newer OSMnx API first
        try:
            # Convert point to the right format (lat, lon) -> (y, x)
            node_id = ox.distance.nearest_nodes(G, point[1], point[0])
            return node_id
        except AttributeError:
            # Fall back to older OSMnx API
            node_id = ox.get_nearest_node(G, point, method='euclidean')
            return node_id
    except Exception as e:
        print(f"Error finding nearest node: {e}")
        
        # Manual fallback - find nearest node by direct calculation
        min_dist = float('inf')
        nearest_node = None
        
        # Convert point to the right format
        lat, lon = point
        
        for node, data in G.nodes(data=True):
            try:
                # Get node coordinates - try different possible attribute names
                node_lat = data.get('y') or data.get('lat')
                node_lon = data.get('x') or data.get('lon')
                
                if node_lat is not None and node_lon is not None:
                    # Calculate Euclidean distance
                    dist = ((lat - node_lat)**2 + (lon - node_lon)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
            except Exception as node_e:
                print(f"Error processing node {node}: {node_e}")
                continue
        
        return nearest_node

def visualize_graph(G, filename='road_network.html'):
    """
    Create a visualization of the road network to verify it's correct.
    """
    # Create a folium map at the center of the graph
    center_lat, center_lon = 31.6295, -7.9811  # Default Marrakech center
    
    try:
        # Get center coordinates from graph
        center_point = ox.geocode('Marrakech, Morocco')
        if isinstance(center_point, tuple) and len(center_point) == 2:
            center_lat, center_lon = center_point
    except:
        pass  # Use default if geocoding fails
        
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Add edges to the map
    for u, v, data in G.edges(data=True):
        # Get node coordinates
        try:
            u_lat = G.nodes[u].get('y') or G.nodes[u].get('lat')
            u_lon = G.nodes[u].get('x') or G.nodes[u].get('lon')
            v_lat = G.nodes[v].get('y') or G.nodes[v].get('lat')
            v_lon = G.nodes[v].get('x') or G.nodes[v].get('lon')
            
            if u_lat and u_lon and v_lat and v_lon:
                # Add edge to map
                folium.PolyLine(
                    locations=[(u_lat, u_lon), (v_lat, v_lon)],
                    color='blue',
                    weight=2,
                    opacity=0.7
                ).add_to(m)
        except Exception as e:
            print(f"Error adding edge {u}-{v} to map: {e}")
            continue
    
    # Add nodes to the map
    for node, data in G.nodes(data=True):
        try:
            lat = data.get('y') or data.get('lat')
            lon = data.get('x') or data.get('lon')
            
            if lat and lon:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2,
                    color='red',
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
        except Exception as e:
            print(f"Error adding node {node} to map: {e}")
            continue
    
    # Save the map
    m.save(filename)
    print(f"Road network visualization saved to {filename}")
    return filename

def find_shortest_path_on_road(G, start_point, end_point, weight='travel_time'):
    """
    Find the shortest path between two points on the road network using actual road segments.
    
    Args:
        G: NetworkX graph representing the road network
        start_point: (lat, lon) tuple for the starting point
        end_point: (lat, lon) tuple for the ending point
        weight: Weight to use for shortest path ('travel_time', 'length', etc.)
        
    Returns:
        path: List of node IDs in the shortest path
        path_length: Total length/time of the path
        path_coords: List of (lat, lon) coordinates for the path following actual roads
    """
    import networkx as nx
    import math
    
    # Input validation
    if not isinstance(start_point, (list, tuple)) or not isinstance(end_point, (list, tuple)):
        print(f"Invalid input format: start_point and end_point must be tuples or lists")
        return None, float('inf'), None
    
    if len(start_point) != 2 or len(end_point) != 2:
        print(f"Invalid coordinates: start_point and end_point must have exactly 2 values")
        return None, float('inf'), None
    
    try:
        # Convert coordinates to float and validate them
        start_lat, start_lon = float(start_point[0]), float(start_point[1])
        end_lat, end_lon = float(end_point[0]), float(end_point[1])
        
        if not (-90 <= start_lat <= 90 and -180 <= start_lon <= 180 and
                -90 <= end_lat <= 90 and -180 <= end_lon <= 180):
            print(f"Warning: Invalid coordinates: start={start_point}, end={end_point}")
            return None, float('inf'), None
        
        # Get the nearest nodes to the start and end points
        try:
            # Try to use distance.nearest_nodes (newer versions of OSMnx)
            start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
            end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)
        except (AttributeError, TypeError):
            # Fall back to get_nearest_node for older versions
            try:
                start_node = ox.get_nearest_node(G, (start_lat, start_lon), method='euclidean')
                end_node = ox.get_nearest_node(G, (end_lat, end_lon), method='euclidean')
            except Exception as e:
                print(f"Error finding nearest nodes using older OSMnx method: {e}")
                # Last resort: try to get any node
                nodes = list(G.nodes())
                if nodes:
                    start_node = nodes[0]
                    end_node = nodes[-1] if len(nodes) > 1 else nodes[0]
                else:
                    print("Graph has no nodes!")
                    return None, float('inf'), None
        
        if start_node is None or end_node is None:
            print(f"Could not find nearest nodes for {start_point} or {end_point}")
            return None, float('inf'), None
        
        # Try to find the shortest path
        try:
            path = nx.shortest_path(G, start_node, end_node, weight=weight)
            
            # Calculate the path length with better error handling
            path_length = 0
            for i in range(len(path)-1):
                try:
                    # Get the edge data and extract the weight
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    
                    # Handle potential multiple edges between nodes
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        # Use the first edge (key 0)
                        if weight in edge_data[0]:
                            path_length += edge_data[0][weight]
                        else:
                            # Try to find any available weight
                            used_weight = next(iter(edge_data[0].keys() & {'length', 'travel_time', 'weight'}), None)
                            if used_weight:
                                path_length += edge_data[0][used_weight]
                            else:
                                # If no weight is found, use a default value
                                path_length += 0.1
                    elif isinstance(edge_data, dict):
                        # Directly use the edge data
                        key = next(iter(edge_data.keys()))
                        if weight in edge_data[key]:
                            path_length += edge_data[key][weight]
                        else:
                            # Try to find any available weight
                            used_weight = next(iter(edge_data[key].keys() & {'length', 'travel_time', 'weight'}), None)
                            if used_weight:
                                path_length += edge_data[key][used_weight]
                            else:
                                # If no weight is found, use a default value
                                path_length += 0.1
                except Exception as e:
                    print(f"Error calculating edge weight for ({path[i]}, {path[i+1]}): {e}")
                    # Use a default small value to continue
                    path_length += 0.1
            
            # Convert length from meters to kilometers if using 'length' weight
            if weight == 'length':
                path_length = path_length / 1000.0  # Convert meters to kilometers
            
            # Ensure path_length is positive and reasonable
            if path_length <= 0:
                # Calculate path length using Haversine formula as a fallback
                path_length = 0
                for i in range(len(path)-1):
                    node1 = path[i]
                    node2 = path[i+1]
                    
                    try:
                        # Get node coordinates
                        y1, x1 = G.nodes[node1]['y'], G.nodes[node1]['x']
                        y2, x2 = G.nodes[node2]['y'], G.nodes[node2]['x']
                        
                        # Calculate Haversine distance
                        R = 6371  # Earth radius in kilometers
                        
                        # Convert to radians
                        lat1, lon1 = math.radians(y1), math.radians(x1)
                        lat2, lon2 = math.radians(y2), math.radians(x2)
                        
                        # Haversine formula
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        distance = R * c
                        
                        path_length += distance
                    except Exception as e:
                        print(f"Error calculating Haversine distance: {e}")
                        # Add a small default distance
                        path_length += 0.1
                
                # If still zero, use a small default value
                if path_length <= 0:
                    path_length = len(path) * 0.1  # Rough estimate
            
            # Get path coordinates
            path_coords = []
            
            # Add starting point
            path_coords.append((start_lat, start_lon))
            
            # Add coordinates along the path
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                
                try:
                    # Get edge data
                    edge_data = G.get_edge_data(u, v)
                    
                    # Handle multiple edges
                    if isinstance(edge_data, dict) and 0 in edge_data:
                        edge_data = edge_data[0]
                    elif isinstance(edge_data, dict):
                        # Get the first edge data
                        key = next(iter(edge_data.keys()))
                        edge_data = edge_data[key]
                    
                    # Check if there's geometry data
                    if 'geometry' in edge_data:
                        # Get detailed coordinates from geometry
                        geom = edge_data['geometry']
                        coords = list(geom.coords)
                        
                        # Add each point of the geometry
                        for x, y in coords:
                            # Convert to lat/lon
                            path_coords.append((y, x))
                    else:
                        # Just add the destination node coordinates
                        y, x = G.nodes[v]['y'], G.nodes[v]['x']
                        path_coords.append((y, x))
                except Exception as e:
                    print(f"Error extracting coordinates for edge ({u}, {v}): {e}")
                    try:
                        # Try to get just the destination node coordinates
                        y, x = G.nodes[v]['y'], G.nodes[v]['x']
                        path_coords.append((y, x))
                    except Exception as e:
                        print(f"Error getting coordinates for node {v}: {e}")
                        # Skip this node
            
            # Add ending point
            path_coords.append((end_lat, end_lon))
            
            # Remove duplicate consecutive points
            unique_coords = []
            for coord in path_coords:
                if not unique_coords or coord != unique_coords[-1]:
                    unique_coords.append(coord)
            
            # Ensure we have at least start and end points
            if len(unique_coords) < 2:
                unique_coords = [(start_lat, start_lon), (end_lat, end_lon)]
            
            return path, path_length, unique_coords
            
        except nx.NetworkXNoPath:
            print(f"No path found between nodes {start_node} and {end_node}")
            # Return a direct path as fallback
            return None, float('inf'), [(start_lat, start_lon), (end_lat, end_lon)]
            
    except Exception as e:
        print(f"Error in find_shortest_path_on_road: {e}")
        return None, float('inf'), None

def create_distance_matrix_on_road(G, location_coords, weight='travel_time'):
    """
    Create a distance matrix for the given locations using the road network.
    
    Args:
        G: NetworkX graph representing the road network
        location_coords: List of (lat, lon) coordinates
        weight: Weight to use for shortest path ('travel_time', 'length', etc.)
        
    Returns:
        distance_matrix: 2D array of distances between locations
        paths_dict: Dictionary containing paths and coordinates for each route
    """
    import time
    import numpy as np
    
    start_time = time.time()
    
    # Input validation and initialization
    if not location_coords or len(location_coords) == 0:
        print("Error: No location coordinates provided")
        return None, None
        
    n = len(location_coords)
    distance_matrix = np.zeros((n, n))
    paths = {}
    
    # First, get all nearest nodes to avoid redundant calculations
    nearest_nodes = []
    for loc in location_coords:
        node = get_nearest_node(G, loc)
        if node is None:
            print(f"Warning: Could not find nearest node for location {loc}")
            # Use a placeholder node ID that will be skipped later
            nearest_nodes.append(-1)
        else:
            nearest_nodes.append(node)
    
    # Calculate paths between all pairs of locations
    for i in range(n):
        if nearest_nodes[i] == -1:
            # Skip invalid nodes
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = 999999
            continue
            
        for j in range(n):
            if i == j:
                # Zero distance for same location
                distance_matrix[i][j] = 0
                continue
                
            if nearest_nodes[j] == -1:
                # Skip invalid nodes
                distance_matrix[i][j] = 999999
                continue
            
            # Check if we already calculated the reverse path (j->i)
            if (j, i) in paths:
                # Reuse the reverse path
                reverse_path, reverse_coords = paths[(j, i)]
                if reverse_path:
                    # Reverse the path and coordinates
                    path = list(reversed(reverse_path))
                    coords = list(reversed(reverse_coords))
                    cost = distance_matrix[j][i]  # Use the same cost
                    distance_matrix[i][j] = cost
                    paths[(i, j)] = (path, coords)
                else:
                    distance_matrix[i][j] = 999999
                continue
            
            # Calculate the path
            try:
                # Use find_road_path instead of find_shortest_path_on_road
                path, cost, coords = find_road_path(G, location_coords[i], location_coords[j], weight)
                if path is None or not coords or len(coords) < 2:
                    # If no path exists or invalid coordinates, set a very large value
                    distance_matrix[i][j] = 999999
                else:
                    # Note: cost is already converted to km in find_road_path if weight is 'length'
                    # Ensure we don't have zero distances by using a minimum value
                    if cost < 0.001:
                        cost = 0.001
                    distance_matrix[i][j] = cost
                    paths[(i, j)] = (path, coords)
            except Exception as e:
                print(f"Error finding path from {i} to {j}: {e}")
                distance_matrix[i][j] = 999999
    
    end_time = time.time()
    print(f"Distance matrix calculation took {end_time - start_time:.2f} seconds")
    
    # IMPORTANT: Do NOT convert to int to preserve decimal precision
    return distance_matrix, paths

def convert_marrakech_graph_to_coordinates(marrakech_graph):
    """
    Convert the Marrakech graph to a dictionary of location names and coordinates.
    
    Args:
        marrakech_graph: NetworkX graph from create_marrakech_sample_graph()
        
    Returns:
        locations_dict: Dictionary of location names and (lat, lon) tuples
    """
    # Get positions from node attributes
    pos = nx.get_node_attributes(marrakech_graph, 'pos')
    
    # Create a dictionary of location names and coordinates
    locations_dict = {}
    for node in marrakech_graph.nodes():
        if node in pos:
            lat, lon = pos[node]
            locations_dict[node] = (lat, lon)
    
    return locations_dict

def find_road_path(G, start_coords, end_coords, weight='length'):
    """
    Enhanced method to find the shortest path between two points on the road network.
    
    Args:
        G: NetworkX graph
        start_coords: (lat, lon) tuple for start
        end_coords: (lat, lon) tuple for end
        weight: 'length' or 'travel_time'
        
    Returns:
        path: List of nodes in the path
        distance: Total path distance/time
        coords: List of (lat, lon) coordinates for the path
    """
    print(f"Finding path from {start_coords} to {end_coords} using weight: {weight}")
    
    # Get the nearest nodes to the start and end points
    start_node = get_nearest_node(G, start_coords)
    end_node = get_nearest_node(G, end_coords)
    
    if start_node is None or end_node is None:
        print("Could not find nearest nodes")
        return None, float('inf'), [start_coords, end_coords]
    
    print(f"Nearest nodes: start={start_node}, end={end_node}")
    
    # Find the shortest path
    try:
        path = nx.shortest_path(G, start_node, end_node, weight=weight)
        print(f"Found path with {len(path)} nodes")
        
        # Calculate the total distance/time
        total = 0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            
            # Check all edges between these nodes (might be multigraph)
            edge_found = False
            min_weight = float('inf')
            
            for edge_key in G[u][v]:
                if weight in G[u][v][edge_key]:
                    edge_found = True
                    edge_weight = G[u][v][edge_key][weight]
                    min_weight = min(min_weight, edge_weight)
            
            if edge_found:
                total += min_weight
            else:
                # If weight not found, use a default value
                print(f"Warning: {weight} not found for edge ({u}, {v})")
                if weight == 'length':
                    # Default 100m if length not found
                    total += 100
                else:
                    # Default 30 seconds if travel_time not found
                    total += 30
        
        # Convert path to coordinates
        coords = []
        
        # Start with the exact starting coordinates
        coords.append(start_coords)
        
        # Add coordinates for each node in the path
        for node in path:
            try:
                lat = G.nodes[node].get('y') or G.nodes[node].get('lat')
                lon = G.nodes[node].get('x') or G.nodes[node].get('lon')
                if lat is not None and lon is not None:
                    coords.append((lat, lon))
            except Exception as e:
                print(f"Error getting coordinates for node {node}: {e}")
                continue
        
        # End with the exact ending coordinates
        coords.append(end_coords)
        
        # Convert distance from meters to kilometers if using 'length'
        if weight == 'length':
            total = total / 1000.0
        elif weight == 'travel_time':
            # Convert seconds to minutes for travel_time
            total = total / 60.0
        
        return path, total, coords
        
    except nx.NetworkXNoPath:
        print(f"No path found between nodes {start_node} and {end_node}")
        return None, float('inf'), [start_coords, end_coords]
    except Exception as e:
        print(f"Error finding path: {e}")
        return None, float('inf'), [start_coords, end_coords]

# Example usage:
"""
# Get the road network
road_G = get_road_network("Marrakech, Morocco", "drive")

# Visualize it to check if it's correct
visualize_graph(road_G, "marrakech_roads.html")

# Find a path between two locations
start_coords = (31.6258, -7.9891)  # Jemaa el-Fnaa
end_coords = (31.6417, -7.9988)    # Majorelle Garden
path, distance, coords = find_road_path(road_G, start_coords, end_coords, "length")

print(f"Path found with {len(path)} nodes, distance: {distance:.2f} km")
print(f"Path coordinates: {len(coords)} points")
"""