import osmnx as ox
import networkx as nx
import numpy as np

# Configure OSMnx
# Update to use settings module directly instead of deprecated config function
try:
    # For newer versions of OSMnx
    ox.settings.use_cache = True
    ox.settings.log_console = False
except AttributeError:
    # For older versions of OSMnx
    ox.config(use_cache=True, log_console=False)

def get_road_network(place_name="Marrakech, Morocco", network_type="drive"):
    """
    Get a road network for a specific place using OSMnx.
    
    Args:
        place_name: Name of the place to get the road network for
        network_type: Type of network to get ('drive', 'bike', 'walk', etc.)
        
    Returns:
        G: NetworkX graph representing the road network
    """
    import os
    import pickle
    from pathlib import Path
    
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create a cache filename based on the place name and network type
    cache_filename = cache_dir / f"{place_name.replace(' ', '_').replace(',', '')}_{network_type}.pkl"
    
    # Check if the cache file exists
    if cache_filename.exists():
        try:
            with open(cache_filename, 'rb') as f:
                G = pickle.load(f)
                print(f"Loaded road network from cache: {cache_filename}")
                return G
        except Exception as e:
            print(f"Error loading cached road network: {e}")
            # Continue to fetch the network if cache loading fails
    
    try:
        # Get the road network
        try:
            # For newer versions of OSMnx
            G = ox.graph_from_place(place_name, network_type=network_type)
        except TypeError:
            # For older versions of OSMnx
            G = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
        
        # Project the graph to UTM
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
        
        # Save the network to cache
        try:
            with open(cache_filename, 'wb') as f:
                pickle.dump(G, f)
                print(f"Saved road network to cache: {cache_filename}")
        except Exception as e:
            print(f"Error saving road network to cache: {e}")
        
        return G
    except Exception as e:
        print(f"Error getting road network: {e}")
        return None

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
        # Get the nearest node
        try:
            # For newer versions of OSMnx
            node_id = ox.distance.nearest_nodes(G, point[1], point[0])
        except AttributeError:
            # For older versions of OSMnx
            node_id = ox.get_nearest_node(G, point, method='euclidean')
        return node_id
    except Exception as e:
        print(f"Error finding nearest node: {e}")
        return None

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
        
        # Get the nearest nodes to the start and end points using network topology
        start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
        end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)
        
        if start_node is None or end_node is None:
            print(f"Could not find nearest nodes for {start_point} or {end_point}")
            return None, float('inf'), None
        
        # Find the shortest path using road network topology
        try:
            path = nx.shortest_path(G, start_node, end_node, weight=weight)
            path_length = nx.shortest_path_length(G, start_node, end_node, weight=weight)
            
            # Get detailed coordinates for the path following road segments
            path_coords = []
            path_coords.append((start_lat, start_lon))  # Add starting point
            
            # Add coordinates for each road segment
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                # Get the geometry of the edge if it exists
                edge_data = G.get_edge_data(u, v)
                if edge_data and 'geometry' in edge_data[0]:
                    # If the edge has geometry data, use it for detailed road shape
                    geom = edge_data[0]['geometry']
                    coords = list(geom.coords)
                    # Convert to lat/lon and add to path
                    for x, y in coords:
                        path_coords.append((y, x))  # Note: swap x,y to lat,lon
                else:
                    # If no geometry, use straight line between nodes
                    try:
                        y = float(G.nodes[v]['y'])
                        x = float(G.nodes[v]['x'])
                        if -90 <= y <= 90 and -180 <= x <= 180:
                            path_coords.append((y, x))
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"Warning: Could not get coordinates for node {v}: {e}")
                        continue
            
            path_coords.append((end_lat, end_lon))  # Add ending point
            
            # Remove duplicate consecutive coordinates while preserving path shape
            path_coords = [coord for i, coord in enumerate(path_coords)
                          if i == 0 or coord != path_coords[i-1]]
            
            return path, path_length, path_coords
            
        except nx.NetworkXNoPath:
            print(f"No path found between nodes {start_node} and {end_node}")
            return None, float('inf'), None
            
    except Exception as e:
        print(f"Error in find_shortest_path_on_road: {e}")
        return None, float('inf'), None
        
        # Calculate the path length
        try:
            # Sum the weights along the path
            path_length = 0
            for i in range(len(path)-1):
                try:
                    # Get the weight for this edge
                    edge_weight = G[path[i]][path[i+1]][0][weight]
                    # Add to total path length
                    path_length += edge_weight
                except (KeyError, TypeError) as e:
                    print(f"Error getting weight for edge ({path[i]}, {path[i+1]}): {e}")
                    # Skip this edge if there's an error
                    continue
            
            # Convert length from meters to kilometers if using 'length' weight
            if weight == 'length':
                path_length = path_length / 1000.0  # Convert meters to kilometers
                
            # Ensure path_length is a reasonable value
            if path_length <= 0 or path_length > 10000:  # Sanity check
                print(f"Warning: Calculated path length {path_length} seems incorrect, using fallback")
                # Fallback: estimate distance based on number of segments
                path_length = len(path) * 0.5  # Rough estimate
        except Exception as e:
            print(f"Error calculating path length: {e}")
            # Fallback to a default weight if the specified one is not available
            fallback_weight = 'length' if weight == 'travel_time' else 'travel_time'
            try:
                path_length = 0
                for i in range(len(path)-1):
                    try:
                        edge_weight = G[path[i]][path[i+1]][0][fallback_weight]
                        path_length += edge_weight
                    except (KeyError, TypeError):
                        # Skip this edge if there's an error
                        continue
                
                # Convert length from meters to kilometers if using 'length' fallback
                if fallback_weight == 'length':
                    path_length = path_length / 1000.0  # Convert meters to kilometers
            except Exception as e:
                print(f"Error calculating path length with fallback weight: {e}")
                # If all else fails, just count the number of edges
                path_length = len(path) * 0.5  # More reasonable estimate (0.5 km per edge)
        
        # Get the path coordinates with more detail for road segments
        path_coords = []
        
        # Validate and add the starting point
        try:
            start_lat, start_lon = float(start_point[0]), float(start_point[1])
            if -90 <= start_lat <= 90 and -180 <= start_lon <= 180:
                path_coords.append((start_lat, start_lon))
            else:
                print(f"Warning: Invalid start point coordinates: {start_point}")
                return None, float('inf'), None
        except (ValueError, TypeError) as e:
            print(f"Error processing start point coordinates: {e}")
            return None, float('inf'), None
        
        # Then add the first node's coordinates
        try:
            start_y, start_x = float(G.nodes[path[0]]['y']), float(G.nodes[path[0]]['x'])
            # Validate coordinates
            if -90 <= start_y <= 90 and -180 <= start_x <= 180:
                # Only add if it's significantly different from the start point
                if abs(start_y - path_coords[0][0]) > 1e-5 or abs(start_x - path_coords[0][1]) > 1e-5:
                    path_coords.append((start_y, start_x))  # (lat, lon)
            else:
                print(f"Warning: Invalid coordinates for node {path[0]}: y={start_y}, x={start_x}")
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing coordinates for node {path[0]}: {e}")
            # Continue without adding this point
        
        # Add intermediate points along each edge in the path
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            
            try:
                # Check if there's geometry data for this edge
                if 'geometry' in G[u][v][0]:
                    # Extract all points from the LineString geometry
                    geom = G[u][v][0]['geometry']
                    # Get all points from the geometry
                    coords = list(geom.coords)
                    
                    if coords:
                        # For the first edge, include all points
                        if i == 0:
                            for point in coords:
                                # Note: point is (x, y) but we need (y, x) for (lat, lon)
                                lat, lon = point[1], point[0]
                                # Validate coordinates before adding
                                if -90 <= lat <= 90 and -180 <= lon <= 180:
                                    path_coords.append((lat, lon))
                        else:
                            # For subsequent edges, skip the first point to avoid duplication
                            # but only if it's very close to the last point we added
                            if path_coords:  # Make sure we have at least one point already
                                last_point = path_coords[-1]
                                first_geom_point = (coords[0][1], coords[0][0])  # Convert to (lat, lon)
                                
                                # Check if the first point of this geometry is already in our path
                                if abs(last_point[0] - first_geom_point[0]) < 1e-6 and abs(last_point[1] - first_geom_point[1]) < 1e-6:
                                    # Skip the first point as it's a duplicate
                                    for point in coords[1:]:
                                        lat, lon = point[1], point[0]
                                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                                            path_coords.append((lat, lon))
                                else:
                                    # Include all points as there's a gap
                                    for point in coords:
                                        lat, lon = point[1], point[0]
                                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                                            path_coords.append((lat, lon))
                            else:
                                # If path_coords is empty, add all points
                                for point in coords:
                                    lat, lon = point[1], point[0]
                                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                                        path_coords.append((lat, lon))
                else:
                    # If no geometry, just add the end node of this segment
                    try:
                        y, x = float(G.nodes[v]['y']), float(G.nodes[v]['x'])
                        if -90 <= y <= 90 and -180 <= x <= 180:
                            path_coords.append((y, x))  # (lat, lon)
                    except (ValueError, TypeError) as e:
                        print(f"Error converting coordinates for node {v}: {e}")
            except (KeyError, AttributeError) as e:
                print(f"Error extracting coordinates for edge ({u}, {v}): {e}")
                # Fallback to just adding the end node
                try:
                    y, x = float(G.nodes[v]['y']), float(G.nodes[v]['x'])
                    if -90 <= y <= 90 and -180 <= x <= 180:
                        path_coords.append((y, x))  # (lat, lon)
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Error processing coordinates for node {v}: {e}")
                    continue
        
        # Add the exact end point coordinates at the end
        # This helps connect the path to the actual destination marker
        # Get the last node's coordinates
        try:
            last_y, last_x = float(G.nodes[path[-1]]['y']), float(G.nodes[path[-1]]['x'])
            # Validate coordinates
            if -90 <= last_y <= 90 and -180 <= last_x <= 180:
                # Only add if it's not already the last point in the path
                if path_coords and (abs(path_coords[-1][0] - last_y) > 1e-5 or abs(path_coords[-1][1] - last_x) > 1e-5):
                    path_coords.append((last_y, last_x))
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing coordinates for end node {path[-1]}: {e}")
        
        # Add the exact end point
        try:
            end_lat, end_lon = float(end_point[0]), float(end_point[1])
            if -90 <= end_lat <= 90 and -180 <= end_lon <= 180:
                # Only add if it's different from the last point we added
                if not path_coords or (abs(path_coords[-1][0] - end_lat) > 1e-5 or abs(path_coords[-1][1] - end_lon) > 1e-5):
                    path_coords.append((end_lat, end_lon))
        except (ValueError, TypeError) as e:
            print(f"Error processing end point coordinates: {e}")
        
        # Remove any duplicate consecutive points while ensuring path continuity
        if len(path_coords) > 1:
            unique_coords = [path_coords[0]]
            for i in range(1, len(path_coords)):
                curr = path_coords[i]
                prev = path_coords[i-1]
                
                # Only add if it's different from the previous point
                # Use a slightly larger threshold to avoid nearly identical points
                if abs(curr[0] - prev[0]) > 1e-6 or abs(curr[1] - prev[1]) > 1e-6:
                    unique_coords.append(curr)
                    
                # If we're at a junction point (where paths might connect),
                # make sure we don't have a large gap to the next point
                elif i < len(path_coords) - 1:
                    next_point = path_coords[i+1]
                    # If there's a significant gap between current and next point,
                    # keep the current point to maintain path continuity
                    if abs(curr[0] - next_point[0]) > 1e-5 or abs(curr[1] - next_point[1]) > 1e-5:
                        unique_coords.append(curr)
            
            path_coords = unique_coords
            
            # Ensure we have at least 2 points to draw a path
            if len(path_coords) < 2:
                print("Warning: Not enough valid coordinates to draw path after filtering")
                # Try to add at least the start and end points
                path_coords = []
                try:
                    start_lat, start_lon = float(start_point[0]), float(start_point[1])
                    end_lat, end_lon = float(end_point[0]), float(end_point[1])
                    if (-90 <= start_lat <= 90 and -180 <= start_lon <= 180 and
                        -90 <= end_lat <= 90 and -180 <= end_lon <= 180):
                        path_coords = [(start_lat, start_lon), (end_lat, end_lon)]
                except (ValueError, TypeError) as e:
                    print(f"Error creating fallback path: {e}")
                    return None, float('inf'), None
        
        return path, path_length, path_coords
    except nx.NetworkXNoPath:
        print(f"No path found between {start_point} and {end_point}")
        return None, float('inf'), None
    except Exception as e:
        print(f"Error finding shortest path: {e}")
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
    # Input validation and initialization
    n = len(location_coords)
    distance_matrix = np.zeros((n, n))
    paths_dict = {}
    
    # Get nearest nodes for all locations once
    nearest_nodes = []
    for lat, lon in location_coords:
        try:
            node = ox.distance.nearest_nodes(G, lon, lat)
            nearest_nodes.append(node)
        except Exception as e:
            print(f"Error finding nearest node for ({lat}, {lon}): {e}")
            return None, None
    # Calculate distances between all pairs of locations
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    # Find shortest path between locations using road network
                    path = nx.shortest_path(G, nearest_nodes[i], nearest_nodes[j], weight=weight)
                    path_length = nx.shortest_path_length(G, nearest_nodes[i], nearest_nodes[j], weight=weight)
                    
                    # Get detailed path coordinates following road segments
                    path_coords = []
                    path_coords.append(location_coords[i])  # Add start point
                    
                    # Add coordinates for each road segment
                    for k in range(len(path)-1):
                        u, v = path[k], path[k+1]
                        # Get the geometry of the edge if it exists
                        edge_data = G.get_edge_data(u, v)
                        if edge_data and 'geometry' in edge_data[0]:
                            # If the edge has geometry data, use it for detailed road shape
                            geom = edge_data[0]['geometry']
                            coords = list(geom.coords)
                            # Convert to lat/lon and add to path
                            for x, y in coords:
                                path_coords.append((y, x))  # Note: swap x,y to lat,lon
                        else:
                            # If no geometry, use straight line between nodes
                            try:
                                y = float(G.nodes[v]['y'])
                                x = float(G.nodes[v]['x'])
                                if -90 <= y <= 90 and -180 <= x <= 180:
                                    path_coords.append((y, x))
                            except (KeyError, ValueError, TypeError) as e:
                                print(f"Warning: Could not get coordinates for node {v}: {e}")
                                continue
                    
                    path_coords.append(location_coords[j])  # Add end point
                    
                    # Remove duplicate consecutive coordinates while preserving path shape
                    path_coords = [coord for k, coord in enumerate(path_coords)
                                  if k == 0 or coord != path_coords[k-1]]
                    
                    # Store the path and its coordinates
                    distance_matrix[i][j] = path_length
                    paths_dict[(i, j)] = (path, path_coords)
                    
                except nx.NetworkXNoPath:
                    print(f"No path found between locations {i} and {j}")
                    distance_matrix[i][j] = float('inf')
                except Exception as e:
                    print(f"Error finding path between locations {i} and {j}: {e}")
                    distance_matrix[i][j] = float('inf')
            else:
                distance_matrix[i][j] = 0
    import time
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
            path, cost, coords = find_shortest_path_on_road(G, location_coords[i], location_coords[j], weight)
            if path is None or not coords or len(coords) < 2:
                # If no path exists or invalid coordinates, set a very large value
                distance_matrix[i][j] = 999999
            else:
                # Note: cost is already converted to km in find_shortest_path_on_road if weight is 'length'
                distance_matrix[i][j] = cost
                paths[(i, j)] = (path, coords)
    
    end_time = time.time()
    print(f"Distance matrix calculation took {end_time - start_time:.2f} seconds")
    
    return distance_matrix.astype(int), paths

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
        lat, lon = pos[node]
        locations_dict[node] = (lat, lon)
    
    return locations_dict
