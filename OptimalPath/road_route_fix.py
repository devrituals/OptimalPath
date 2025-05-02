import folium
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Radius of Earth in kilometers
    
    return distance


def get_nearest_node_fixed(G, point):
    """
    Get the nearest node in the graph to a point.
    Robust version that handles errors with CRS and attribute names.
    
    Args:
        G: NetworkX graph
        point: (lat, lon) tuple
        
    Returns:
        node_id: ID of the nearest node
    """
    try:
        # Manual calculation to find nearest node
        min_dist = float('inf')
        nearest_node = None
        
        # Get coordinates for the point
        try:
            lat, lon = float(point[0]), float(point[1])
        except (ValueError, TypeError, IndexError):
            print(f"Invalid point coordinates: {point}")
            return None
        
        # Check each node
        for node, data in G.nodes(data=True):
            try:
                # Try different coordinate attribute names
                node_lat = None
                node_lon = None
                
                if 'y' in data:
                    node_lat = data['y']
                elif 'lat' in data:
                    node_lat = data['lat']
                elif 'pos' in data and isinstance(data['pos'], (list, tuple)) and len(data['pos']) >= 2:
                    node_lat = data['pos'][0]
                
                if 'x' in data:
                    node_lon = data['x']
                elif 'lon' in data:
                    node_lon = data['lon']
                elif 'pos' in data and isinstance(data['pos'], (list, tuple)) and len(data['pos']) >= 2:
                    node_lon = data['pos'][1]
                
                # Skip if we can't find coordinates
                if node_lat is None or node_lon is None:
                    continue
                
                # Calculate Euclidean distance
                dist = ((lat - node_lat)**2 + (lon - node_lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            except Exception as e:
                continue
        
        return nearest_node
        
    except Exception as e:
        print(f"Error finding nearest node: {e}")
        return None


def find_road_path_fixed(G, start_coords, end_coords, weight='length'):
    """
    Find a path between two points following roads.
    Robust version that handles errors with the graph and coordinates.
    
    Args:
        G: NetworkX graph representing the road network
        start_coords: (lat, lon) tuple for the starting point
        end_coords: (lat, lon) tuple for the ending point
        weight: Weight to use for shortest path ('length' or 'travel_time')
        
    Returns:
        path: List of node IDs in the shortest path
        distance: Total distance/time of the path
        coords: List of (lat, lon) coordinates for the path
    """
    print(f"Finding path from {start_coords} to {end_coords} using weight: {weight}")
    
    # Input validation
    if not isinstance(start_coords, (list, tuple)) or not isinstance(end_coords, (list, tuple)):
        print("Invalid coordinates format")
        return None, 0, [start_coords, end_coords]
    
    try:
        # Get the nearest nodes to the start and end points
        start_node = get_nearest_node_fixed(G, start_coords)
        end_node = get_nearest_node_fixed(G, end_coords)
        
        if start_node is None or end_node is None:
            print("Could not find nearest nodes")
            return None, 0, [start_coords, end_coords]
        
        print(f"Nearest nodes: start={start_node}, end={end_node}")
        
        # Find the shortest path
        try:
            path = nx.shortest_path(G, start_node, end_node, weight=weight)
            print(f"Found path with {len(path)} nodes")
            
            # Calculate the total distance
            total = 0
            for i in range(len(path)-1):
                try:
                    # Try to get weight from edge data
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    
                    # Handle case where there might be multiple edges or missing weight
                    if isinstance(edge_data, dict):
                        # For multigraph, find the minimum weight
                        min_weight = float('inf')
                        weight_found = False
                        
                        # Check all possible edges between these nodes
                        for edge_key, edge_attrs in edge_data.items():
                            if weight in edge_attrs:
                                weight_found = True
                                min_weight = min(min_weight, edge_attrs[weight])
                        
                        if weight_found:
                            total += min_weight
                        else:
                            # Fall back to direct distance
                            try:
                                node1_coords = get_node_coords(G, path[i])
                                node2_coords = get_node_coords(G, path[i+1])
                                if node1_coords and node2_coords:
                                    dist = calculate_haversine_distance(
                                        node1_coords[0], node1_coords[1],
                                        node2_coords[0], node2_coords[1]
                                    )
                                    total += dist * 1000 if weight == 'length' else dist * 60
                                else:
                                    total += 0.1  # Default small value
                            except:
                                total += 0.1  # Default small value
                    else:
                        # Fall back to direct distance
                        try:
                            node1_coords = get_node_coords(G, path[i])
                            node2_coords = get_node_coords(G, path[i+1])
                            if node1_coords and node2_coords:
                                dist = calculate_haversine_distance(
                                    node1_coords[0], node1_coords[1],
                                    node2_coords[0], node2_coords[1]
                                )
                                total += dist * 1000 if weight == 'length' else dist * 60
                            else:
                                total += 0.1  # Default small value
                        except:
                            total += 0.1  # Default small value
                except Exception as e:
                    print(f"Error calculating edge weight: {e}")
                    total += 0.1  # Default small value
            
            # Create path coordinates
            path_coords = []
            
            # Add the start coordinates
            path_coords.append(start_coords)
            
            # Add coordinates for each node in the path
            for node in path:
                try:
                    coords = get_node_coords(G, node)
                    if coords:
                        path_coords.append(coords)
                except Exception as e:
                    print(f"Error getting node coordinates: {e}")
            
            # Add the end coordinates
            path_coords.append(end_coords)
            
            # Convert from meters to kilometers or seconds to minutes
            if weight == 'length':
                total = total / 1000.0
            elif weight == 'travel_time':
                total = total / 60.0
            
            # Ensure we don't return zero
            if total <= 0:
                total = 0.001
                
            return path, total, path_coords
            
        except nx.NetworkXNoPath:
            print(f"No path found between nodes {start_node} and {end_node}")
            return None, 0, [start_coords, end_coords]
        except Exception as e:
            print(f"Error finding path: {e}")
            return None, 0, [start_coords, end_coords]
            
    except Exception as e:
        print(f"Error in find_road_path: {e}")
        return None, 0, [start_coords, end_coords]


def get_node_coords(G, node):
    """Get coordinates for a node trying different attribute names"""
    try:
        data = G.nodes[node]
        
        # Try different coordinate attribute names
        if 'y' in data and 'x' in data:
            return (data['y'], data['x'])
        elif 'lat' in data and 'lon' in data:
            return (data['lat'], data['lon'])
        elif 'pos' in data and isinstance(data['pos'], (list, tuple)) and len(data['pos']) >= 2:
            return (data['pos'][0], data['pos'][1])
        
        return None
    except:
        return None


def draw_route_on_map(m, route_coords, color='blue', weight=5, opacity=0.7, route_name="Route"):
    """
    Draw a route on a folium map using PolyLine.
    
    Args:
        m: Folium map
        route_coords: List of (lat, lon) coordinates
        color: Line color
        weight: Line weight
        opacity: Line opacity
        route_name: Name of the route
        
    Returns:
        m: Updated map
    """
    # Validate coordinates
    valid_coords = []
    for coord in route_coords:
        if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
            if coord[0] is not None and coord[1] is not None:
                try:
                    lat, lon = float(coord[0]), float(coord[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        valid_coords.append((lat, lon))
                except:
                    continue
    
    # Only draw if we have at least 2 valid coordinates
    if len(valid_coords) >= 2:
        # Add the route line
        folium.PolyLine(
            locations=valid_coords, 
            color=color, 
            weight=weight, 
            opacity=opacity,
            tooltip=route_name
        ).add_to(m)
        
        # Calculate and display distance
        try:
            distance = 0
            for i in range(len(valid_coords)-1):
                lat1, lon1 = valid_coords[i]
                lat2, lon2 = valid_coords[i+1]
                distance += calculate_haversine_distance(lat1, lon1, lat2, lon2)
            
            # Add a distance label at the midpoint
            midpoint_idx = len(valid_coords) // 2
            if midpoint_idx > 0:
                folium.Marker(
                    location=valid_coords[midpoint_idx],
                    icon=folium.DivIcon(
                        icon_size=(100, 20),
                        icon_anchor=(50, 10),
                        html=f'<div style="font-size: 10pt; color: {color}; background-color: white; padding: 2px 5px; border: 1px solid {color}; border-radius: 3px; text-align: center;">{distance:.2f} km</div>'
                    )
                ).add_to(m)
        except Exception as e:
            print(f"Error calculating route distance: {e}")
    
    return m
