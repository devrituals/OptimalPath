import folium
from folium.features import DivIcon
import networkx as nx
import osmnx as ox

def create_road_folium_map(road_graph=None, path_coords=None, markers=None, path_nodes=None):
    """
    Create a Folium map with a road-based path.
    
    Args:
        road_graph: NetworkX graph representing the road network
        path_coords: List of (lat, lon) coordinates for the path
        markers: List of (lat, lon, name) tuples for markers
        path_nodes: List of node IDs in the path
        
    Returns:
        m: Folium map
    """
    # If no road graph is provided, create a default map centered on Marrakech
    if road_graph is None and path_coords is None and markers is None:
        # Default center of Marrakech
        center_lat, center_lon = 31.6295, -7.9811
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
        return m
    
    # If we have path coordinates, find the center of the map
    if path_coords and len(path_coords) > 0:
        # Filter out any None values that might have crept in
        valid_coords = [(lat, lon) for lat, lon in path_coords if lat is not None and lon is not None]
        if valid_coords:
            center_lat = sum(lat for lat, _ in valid_coords) / len(valid_coords)
            center_lon = sum(lon for _, lon in valid_coords) / len(valid_coords)
        else:
            # Default center of Marrakech
            center_lat, center_lon = 31.6295, -7.9811
    elif markers and len(markers) > 0:
        center_lat = sum(lat for lat, _, _ in markers) / len(markers)
        center_lon = sum(lon for _, lon, _ in markers) / len(markers)
    else:
        # Default center of Marrakech
        center_lat, center_lon = 31.6295, -7.9811
    
    # Create a map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    
    # Add markers if provided
    if markers:
        for i, (lat, lon, name) in enumerate(markers):
            # Use different styling for delivery person vs stops
            if i == 0 and "Delivery Person" in name:
                # Delivery person marker with distinctive icon
                folium.Marker(
                    location=[lat, lon],
                    popup=name,
                    icon=folium.Icon(color='green', icon='user', prefix='fa')
                ).add_to(m)
            else:
                # Regular stop marker
                folium.Marker(
                    location=[lat, lon],
                    popup=name,
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
    
    # Add the path if provided
    if path_coords and len(path_coords) > 1:
        # Filter out any None values and validate coordinates
        valid_coords = []
        for lat, lon in path_coords:
            if lat is not None and lon is not None:
                try:
                    lat_float = float(lat)
                    lon_float = float(lon)
                    if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                        valid_coords.append((lat_float, lon_float))
                    else:
                        print(f"Warning: Invalid coordinate values: lat={lat_float}, lon={lon_float}")
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert coordinates to float: lat={lat}, lon={lon}")
        
        # Debug information
        print(f"Path coordinates: {len(path_coords)}, Valid coordinates: {len(valid_coords)}")
        
        # Ensure we have at least 2 valid coordinates to draw a path
        if len(valid_coords) < 2:
            print("Warning: Not enough valid coordinates to draw path")
            # Try to recover by using node coordinates from path_nodes if available
            if road_graph is not None and path_nodes is not None and len(path_nodes) >= 2:
                print("Attempting to recover path using node coordinates...")
                valid_coords = []
                for node in path_nodes:
                    try:
                        y, x = float(road_graph.nodes[node]['y']), float(road_graph.nodes[node]['x'])
                        if -90 <= y <= 90 and -180 <= x <= 180:
                            valid_coords.append((y, x))
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"Error extracting coordinates for node {node}: {e}")
                        continue
                
                # If we still don't have enough valid coordinates, try to use markers
                if len(valid_coords) < 2 and markers and len(markers) >= 2:
                    print("Attempting to recover path using marker coordinates...")
                    valid_coords = []
                    for lat, lon, _ in markers:
                        try:
                            lat_float = float(lat)
                            lon_float = float(lon)
                            if -90 <= lat_float <= 90 and -180 <= lon_float <= 180:
                                valid_coords.append((lat_float, lon_float))
                        except (ValueError, TypeError):
                            continue
        
        if len(valid_coords) > 1:
            # Create a polyline for the path
            # Use a more visible style for the main route
            folium.PolyLine(
                locations=valid_coords,
                color='red',
                weight=7,  # Increase thickness for better visibility
                opacity=1.0,
                popup="Delivery Route",
                tooltip="Delivery Route",  # Add tooltip for better UX
                smooth_factor=1.2  # Adjust smooth factor for better path rendering
            ).add_to(m)

            
            # Add a second polyline with a different style to create a border effect
            # This makes the route more visible against different backgrounds
            folium.PolyLine(
                locations=valid_coords,
                color='white',
                weight=3,  # Thinner than the main line
                opacity=0.7,
                popup="Delivery Route",
                smooth_factor=1.2
            ).add_to(m)
            
            # Debug information about the path
            print(f"Drawing path with {len(valid_coords)} points")
            if len(valid_coords) > 1:
                print(f"First point: {valid_coords[0]}, Last point: {valid_coords[-1]}")
                
            # Add markers at key points in the path
            # This helps visualize the actual route more clearly
            # Only add markers at the start, end, and some intermediate points to avoid clutter
            key_points = []
            if len(valid_coords) > 0:
                key_points.append(0)  # Start point
            if len(valid_coords) > 1:
                key_points.append(len(valid_coords) - 1)  # End point
            
            # Add some intermediate points
            if len(valid_coords) > 10:
                step = len(valid_coords) // 5  # Add about 5 intermediate points
                for i in range(step, len(valid_coords) - 1, step):
                    key_points.append(i)
            
            # Add small markers at key points
            for i in key_points:
                lat, lon = valid_coords[i]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color='blue',
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"Path point {i}"
                ).add_to(m)
            
            # Add distance information to the map
            if len(valid_coords) > 1:
                # Calculate approximate distance along the path
                from math import radians, sin, cos, sqrt, atan2
                
                total_distance = 0
                valid_segments = 0
                
                for i in range(len(valid_coords)-1):
                    lat1, lon1 = valid_coords[i]
                    lat2, lon2 = valid_coords[i+1]
                    
                    try:
                        # Convert latitude and longitude from degrees to radians
                        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
                        
                        # Haversine formula
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1-a))
                        distance = 6371 * c  # Distance in kilometers (Earth radius = 6371 km)
                        
                        # Only add reasonable distances (avoid extremely large jumps)
                        if 0 < distance < 100:  # Cap at 100km for a single segment and ensure positive
                            total_distance += distance
                            valid_segments += 1
                            print(f"Segment {i}: {distance:.3f} km")
                    except (ValueError, TypeError) as e:
                        print(f"Error calculating distance for segment {i}: {e}")
                        continue
                
                # Ensure total_distance is a reasonable value
                if total_distance <= 0 or total_distance > 10000 or valid_segments == 0:  # Sanity check
                    print(f"Warning: Calculated distance {total_distance} seems incorrect, using fallback")
                    # Fallback: estimate distance based on number of segments
                    total_distance = len(valid_coords) * 0.5  # Rough estimate
                else:
                    # Calculate average segment distance for more accurate reporting
                    avg_segment_distance = total_distance / valid_segments
                    print(f"Average segment distance: {avg_segment_distance:.3f} km")
                    print(f"Total calculated distance: {total_distance:.3f} km over {valid_segments} segments")
                
                # Add a legend with the total distance
                # Format the distance with appropriate precision based on value
                if total_distance < 10:
                    # For shorter distances, show 3 decimal places
                    distance_display = f"{total_distance:.3f}"
                else:
                    # For longer distances, 2 decimal places is sufficient
                    distance_display = f"{total_distance:.2f}"
                    
                legend_html = f'''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 180px; height: 90px; 
                            border:2px solid grey; z-index:9999; font-size:14px;
                            background-color:white; padding: 8px;
                            border-radius: 5px;">
                    <b>Route Information</b><br>
                    Total Distance: {distance_display} km<br>
                    Stops: {len(markers)-1 if markers else 0}
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add path from delivery person to first stop with a different color
            if markers and len(markers) >= 2 and "Delivery Person" in markers[0][2]:
                # Try to find the first stop coordinates in the path
                first_stop_lat, first_stop_lon = markers[1][0], markers[1][1]
                first_stop_index = -1
                
                # Find the closest point in the path to the first stop
                min_distance = float('inf')
                for i, (lat, lon) in enumerate(valid_coords):
                    dist = ((lat - first_stop_lat)**2 + (lon - first_stop_lon)**2)**0.5
                    if dist < min_distance:
                        min_distance = dist
                        first_stop_index = i
                
                # If we found a reasonable match, use it
                if first_stop_index >= 0 and first_stop_index < len(valid_coords):
                    first_stop_coords = valid_coords[:first_stop_index+1]
                else:
                    # Fallback: use the first few points (better than nothing)
                    first_stop_coords = valid_coords[:min(5, len(valid_coords))]
                
                if first_stop_coords and len(first_stop_coords) > 1:
                    # Filter out any None values
                    valid_first_stop_coords = [(lat, lon) for lat, lon in first_stop_coords if lat is not None and lon is not None]
                    
                    if len(valid_first_stop_coords) > 1:
                        folium.PolyLine(
                            locations=valid_first_stop_coords,
                            color='green',
                            weight=4,
                            opacity=0.8,
                            popup="Path to first stop"
                        ).add_to(m)
                        
                        # Add an arrow to indicate direction
                        arrow_position = valid_first_stop_coords[len(valid_first_stop_coords)//2]  # Middle of the path
                        folium.Marker(
                            location=arrow_position,
                            icon=folium.Icon(color='green', icon='arrow-right', prefix='fa'),
                            popup="Direction to first stop"
                        ).add_to(m)
        else:
            print("Warning: Not enough valid coordinates to draw path")
        
        # Add numbered markers for each stop in the route
        if markers and len(markers) >= 2:
            # Add numbered markers for each stop (excluding delivery person)
            for i in range(1, len(markers)):
                # Create a circle marker with the stop number
                folium.CircleMarker(
                    location=[markers[i][0], markers[i][1]],
                    radius=18,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.9,
                    popup=f"Stop {i}: {markers[i][2]}"
                ).add_to(m)
                
                # Add the stop number as text
                folium.Marker(
                    location=[markers[i][0], markers[i][1]],
                    icon=DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-align: center;">{i}</div>'
                    )
                ).add_to(m)
            
            # Add start marker (first stop)
            folium.Marker(
                location=[markers[1][0], markers[1][1]],  # First stop (index 1) after delivery person
                icon=DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 12pt; color: red; font-weight: bold;">Start: {markers[1][2]}</div>'
                )
            ).add_to(m)
            
            # Add end marker (last stop)
            folium.Marker(
                location=[markers[-1][0], markers[-1][1]],
                icon=DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 12pt; color: red; font-weight: bold;">End: {markers[-1][2]}</div>'
                )
            ).add_to(m)
    
    # If we have a road graph and path nodes, we can add all roads for context
    # But only add a limited number to avoid performance issues
    if road_graph is not None and path_nodes is not None:
        # Create a set of path edges for faster lookup
        path_edges = set()
        if path_nodes and len(path_nodes) > 1:
            for i in range(len(path_nodes) - 1):
                path_edges.add((path_nodes[i], path_nodes[i+1]))
                path_edges.add((path_nodes[i+1], path_nodes[i]))  # Add both directions
                # Add edges with keys for more robust matching
                for k in range(3):  # Try a few common key values
                    path_edges.add((path_nodes[i], path_nodes[i+1], k))
                    path_edges.add((path_nodes[i+1], path_nodes[i], k))
            
            # Debug information
            print(f"Path nodes: {len(path_nodes)}, Path edges: {len(path_edges)}")
            print(f"First few path nodes: {path_nodes[:min(5, len(path_nodes))]}")
            print(f"First few path edges: {list(path_edges)[:min(5, len(path_edges))]}")
        
        # Only add edges that are part of the path or very close to it
        # This improves performance by not rendering the entire road network
        for u, v, data in road_graph.edges(data=True):
            try:
                # Check if this edge is part of the path
                # Use a more robust check that handles different edge key formats
                is_path_edge = False
                
                # Check all possible combinations of edge representations
                if (u, v) in path_edges or (v, u) in path_edges:
                    is_path_edge = True
                    # Debug information for path edges
                    print(f"Found path edge: ({u}, {v})")
                # Some graphs might use edge keys with additional data
                elif any((u, v, k) in path_edges or (v, u, k) in path_edges for k in range(10)):
                    is_path_edge = True
                    # Debug information for path edges with keys
                    print(f"Found path edge with key: ({u}, {v}, k)")
                # Additional check for edges that might be part of the path but not exactly matching
                elif path_nodes and u in path_nodes and v in path_nodes:
                    # Check if these nodes are adjacent in the path
                    for i in range(len(path_nodes) - 1):
                        if (path_nodes[i] == u and path_nodes[i+1] == v) or \
                           (path_nodes[i] == v and path_nodes[i+1] == u):
                            is_path_edge = True
                            print(f"Found path edge through node adjacency: ({u}, {v})")
                            break
                
                # Only add path edges and a small subset of nearby edges
                # Print debug info for the first few path edges to verify they're being detected
                if is_path_edge:
                    # Debug the first few path edges
                    if len(path_edges) > 0 and (u, v) in list(path_edges)[:5]:
                        print(f"Rendering path edge: ({u}, {v})")
                    edge_color = 'red'
                    edge_weight = 6  # Increase weight for better visibility
                    edge_opacity = 1.0
                    
                    # Add a small marker at each node in the path to help visualize connectivity
                    folium.CircleMarker(
                        location=[road_graph.nodes[u]['y'], road_graph.nodes[u]['x']],
                        radius=3,
                        color='darkred',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(m)
                    
                    # Check if there's geometry data for this edge
                    if 'geometry' in data:
                        # Extract all points from the LineString geometry
                        geom = data['geometry']
                        # Convert the geometry to a list of (lat, lon) coordinates
                        edge_coords = [(point[1], point[0]) for point in list(geom.coords)]
                        
                        # Add the edge to the map with detailed geometry
                        folium.PolyLine(
                            locations=edge_coords,
                            color=edge_color,
                            weight=edge_weight,
                            opacity=edge_opacity
                        ).add_to(m)
                    else:
                        # If no geometry, just use the node coordinates
                        u_y, u_x = road_graph.nodes[u]['y'], road_graph.nodes[u]['x']
                        v_y, v_x = road_graph.nodes[v]['y'], road_graph.nodes[v]['x']
                        
                        # Add the edge to the map
                        folium.PolyLine(
                            locations=[(u_y, u_x), (v_y, v_x)],
                            color=edge_color,
                            weight=edge_weight,
                            opacity=edge_opacity
                        ).add_to(m)
                elif path_nodes:  # Add nearby roads for context with different styling
                    # Get the coordinates of the nodes
                    u_y, u_x = road_graph.nodes[u]['y'], road_graph.nodes[u]['x']
                    v_y, v_x = road_graph.nodes[v]['y'], road_graph.nodes[v]['x']
                    
                    # Check if this edge is close to the path (within a certain distance)
                    # This helps provide context without rendering the entire network
                    is_nearby = False
                    for node in path_nodes:
                        # Calculate distance to path nodes
                        if node in road_graph.nodes:
                            node_y, node_x = road_graph.nodes[node]['y'], road_graph.nodes[node]['x']
                            # Simple distance check (can be optimized)
                            dist_u = ((u_y - node_y)**2 + (u_x - node_x)**2)**0.5
                            dist_v = ((v_y - node_y)**2 + (v_x - node_x)**2)**0.5
                            
                            # If either node of the edge is close to a path node
                            # Increase the distance threshold to include more context roads
                            if dist_u < 0.002 or dist_v < 0.002:  # Approximately 200m
                                is_nearby = True
                                break
                    
                    if is_nearby:
                        # Check if there's geometry data for this edge
                        if 'geometry' in data:
                            # Extract all points from the LineString geometry
                            geom = data['geometry']
                            # Convert the geometry to a list of (lat, lon) coordinates
                            edge_coords = [(point[1], point[0]) for point in list(geom.coords)]
                            
                            # Add the edge to the map with detailed geometry
                            folium.PolyLine(
                                locations=edge_coords,
                                color='gray',
                                weight=2,
                                opacity=0.5
                            ).add_to(m)
                        else:
                            # If no geometry, just use the node coordinates
                            # Add nearby roads with different styling
                            folium.PolyLine(
                                locations=[(u_y, u_x), (v_y, v_x)],
                                color='gray',
                                weight=2,
                                opacity=0.5
                            ).add_to(m)
            except KeyError:
                # Skip edges with missing coordinates
                continue
    
    return m

def plot_route_on_map(road_graph, path, locations_dict=None, location_names=None):
    """
    Plot a route on a Folium map.
    
    Args:
        road_graph: NetworkX graph representing the road network
        path: List of node IDs in the path
        locations_dict: Dictionary of location names and (lat, lon) tuples
        location_names: List of location names in the path
        
    Returns:
        m: Folium map
    """
    # Get the path coordinates
    path_coords = []
    for node in path:
        # Get the coordinates of the node
        y, x = road_graph.nodes[node]['y'], road_graph.nodes[node]['x']
        path_coords.append((y, x))  # (lat, lon)
    
    # Create markers for the locations
    markers = []
    if locations_dict and location_names:
        for name in location_names:
            lat, lon = locations_dict[name]
            markers.append((lat, lon, name))
    
    # Create the map
    m = create_road_folium_map(road_graph, path_coords, markers, path)
    
    return m
