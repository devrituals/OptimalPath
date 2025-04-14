import folium
from folium.features import DivIcon
import networkx as nx
import traceback
from math import radians, sin, cos, sqrt, atan2

# Try to import osmnx but handle its absence gracefully
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    print("Warning: osmnx package is not available. Some map features will be limited.")
    OSMNX_AVAILABLE = False

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
    try:
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
                if lat is None or lon is None:
                    print(f"Warning: Skipping marker with None coordinates: {name}")
                    continue
                    
                try:
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
                except Exception as e:
                    print(f"Error adding marker {name}: {e}")
        
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
            
            print(f"Path coordinates: {len(path_coords)}, Valid coordinates: {len(valid_coords)}")
            
            # Handle case when we don't have enough valid coordinates
            if len(valid_coords) < 2:
                print("Warning: Not enough valid coordinates to draw path. Attempting recovery...")
                
                # Try to recover path using road network nodes if available
                if road_graph is not None and path_nodes is not None and len(path_nodes) >= 2:
                    print("Attempting to recover path using node coordinates...")
                    valid_coords = []
                    
                    # Try to extract complete geometries from the road graph
                    try:
                        # Extract geometry for every edge in the path
                        edge_geometries = []
                        for i in range(len(path_nodes) - 1):
                            u, v = path_nodes[i], path_nodes[i+1]
                            if road_graph.has_edge(u, v):
                                edge_data = road_graph.get_edge_data(u, v, 0)  # 0 is the default key
                                if 'geometry' in edge_data and edge_data['geometry'] is not None:
                                    # Convert shapely LineString to list of coordinates
                                    coords = [(point[1], point[0]) for point in edge_data['geometry'].coords]
                                    edge_geometries.append(coords)
                                else:
                                    # If no geometry, use node coordinates
                                    try:
                                        u_coords = (road_graph.nodes[u]['y'], road_graph.nodes[u]['x'])
                                        v_coords = (road_graph.nodes[v]['y'], road_graph.nodes[v]['x'])
                                        edge_geometries.append([u_coords, v_coords])
                                    except KeyError:
                                        print(f"Error: Missing coordinate data for nodes {u} or {v}")
                        
                        # Combine all edge geometries into a single path
                        if edge_geometries:
                            for geom in edge_geometries:
                                valid_coords.extend(geom)
                            
                            # Remove duplicate consecutive points
                            valid_coords = [coord for i, coord in enumerate(valid_coords)
                                          if i == 0 or coord != valid_coords[i-1]]
                        else:
                            print("No edge geometries found, falling back to node coordinates")
                    except Exception as e:
                        print(f"Error extracting edge geometries: {e}")
                        traceback.print_exc()
                    
                    # If still no valid coords, extract node coordinates directly
                    if len(valid_coords) < 2:
                        for node in path_nodes:
                            try:
                                if node in road_graph.nodes:
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
            
            # Draw the path
            if len(valid_coords) > 1:
                try:
                    # Draw the path as a polyline
                    folium.PolyLine(
                        locations=valid_coords,
                        color='red',
                        weight=5,
                        opacity=1.0,
                        popup="Route",
                        tooltip="Route",
                        smooth_factor=1.2
                    ).add_to(m)
                    
                    # Add a border to make the path more visible
                    folium.PolyLine(
                        locations=valid_coords,
                        color='white',
                        weight=8,
                        opacity=0.4,
                        smooth_factor=1.2
                    ).add_to(m)
                    
                    # Calculate and display the path distance
                    total_distance = calculate_path_distance(valid_coords)
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 50px; left: 50px; width: 180px; height: 90px; 
                                border:2px solid grey; z-index:9999; font-size:14px;
                                background-color:white; padding: 8px;
                                border-radius: 5px;">
                        <b>Route Information</b><br>
                        Total Distance: {total_distance:.2f} km<br>
                        Stops: {len(markers)-1 if markers else 0}
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(legend_html))
                
                except Exception as e:
                    print(f"Error adding path to map: {e}")
                    traceback.print_exc()
            else:
                print("Warning: Not enough valid coordinates to draw path")
            
            # Add numbered markers for each stop
            if markers and len(markers) >= 2:
                try:
                    # Add numbered markers for each stop (excluding delivery person)
                    for i in range(1, len(markers)):
                        # Skip markers with None coordinates
                        if markers[i][0] is None or markers[i][1] is None:
                            continue
                            
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
                except Exception as e:
                    print(f"Error adding numbered markers: {e}")
        
        # Add road network visualization if we have path_nodes
        if OSMNX_AVAILABLE and road_graph is not None and path_nodes is not None and len(path_nodes) > 1:
            try:
                # Create a list of edges that form the path
                path_edges = []
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i+1]
                    # Check if this edge exists in the graph
                    if road_graph.has_edge(u, v):
                        path_edges.append((u, v, 0))  # 0 is the default key for MultiDiGraph
                    elif road_graph.has_edge(v, u):
                        path_edges.append((v, u, 0))  # Check reverse direction
                
                if path_edges:
                    # Extract the subgraph containing only the path edges
                    edge_colors = ['red'] * len(path_edges)
                    
                    # Get style parameters for the road network
                    # We're using osmnx just for the styling function
                    style_kwargs = {
                        'edge_color': edge_colors,
                        'edge_linewidth': 5,
                        'edge_opacity': 0.8,
                        'node_size': 0,  # Don't show nodes
                    }
                    
                    # Add each edge with its geometry to the map
                    for edge in path_edges:
                        u, v, k = edge
                        data = road_graph.get_edge_data(u, v, k)
                        
                        if 'geometry' in data and data['geometry'] is not None:
                            # Convert shapely LineString to list of coordinates
                            coords = [(point[1], point[0]) for point in data['geometry'].coords]
                            
                            # Add the edge to the map
                            folium.PolyLine(
                                locations=coords,
                                color='red',
                                weight=5,
                                opacity=0.8
                            ).add_to(m)
                        else:
                            # If no geometry, use node coordinates
                            try:
                                u_coords = (road_graph.nodes[u]['y'], road_graph.nodes[u]['x'])
                                v_coords = (road_graph.nodes[v]['y'], road_graph.nodes[v]['x'])
                                
                                folium.PolyLine(
                                    locations=[u_coords, v_coords],
                                    color='red',
                                    weight=5,
                                    opacity=0.8
                                ).add_to(m)
                            except KeyError:
                                print(f"Error: Missing coordinate data for nodes {u} or {v}")
            except Exception as e:
                print(f"Error visualizing road network: {e}")
                traceback.print_exc()
        
        return m
    
    except Exception as e:
        print(f"Error creating road map: {e}")
        traceback.print_exc()
        # Return a default map as fallback
        default_m = folium.Map(location=[31.6295, -7.9811], zoom_start=14, tiles="OpenStreetMap")
        return default_m

def calculate_path_distance(coords):
    """
    Calculate the total distance of a path using haversine distance
    
    Args:
        coords: List of (lat, lon) coordinates
    
    Returns:
        total_distance: Total distance in kilometers
    """
    total_distance = 0
    valid_segments = 0
    
    for i in range(len(coords)-1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]
        
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
        except Exception as e:
            print(f"Error calculating distance for segment {i}: {e}")
    
    # Ensure total_distance is a reasonable value
    if total_distance <= 0 or total_distance > 10000 or valid_segments == 0:
        # Fallback: estimate distance based on number of segments
        total_distance = len(coords) * 0.5  # Rough estimate
    
    return total_distance

def plot_route_on_map(map_obj, path_nodes, road_graph, popup_text=None):
    """
    Plot a route from a list of nodes on an existing map
    
    Args:
        map_obj: Folium map object
        path_nodes: List of node IDs in the path
        road_graph: NetworkX graph representing the road network
        popup_text: Text to show in popup (optional)
    """
    try:
        if not path_nodes or len(path_nodes) < 2:
            print("Route has less than 2 points, cannot plot")
            return
            
        # Add each edge in the path with its geometry to the map
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            
            # Check if this edge exists in the graph
            if road_graph.has_edge(u, v):
                edge = (u, v, 0)  # 0 is the default key for MultiDiGraph
            elif road_graph.has_edge(v, u):
                edge = (v, u, 0)  # Check reverse direction
            else:
                # Edge doesn't exist, draw direct line between nodes
                try:
                    u_coords = (road_graph.nodes[u]['y'], road_graph.nodes[u]['x'])
                    v_coords = (road_graph.nodes[v]['y'], road_graph.nodes[v]['x'])
                    
                    folium.PolyLine(
                        locations=[u_coords, v_coords],
                        color='red',
                        weight=4,
                        opacity=0.7,
                        popup=popup_text if popup_text else "Route"
                    ).add_to(map_obj)
                except KeyError:
                    print(f"Error: Missing coordinate data for nodes {u} or {v}")
                continue
            
            # Get edge data
            data = road_graph.get_edge_data(*edge)
            
            if 'geometry' in data and data['geometry'] is not None:
                # Convert shapely LineString to list of coordinates
                coords = [(point[1], point[0]) for point in data['geometry'].coords]
                
                # Add the edge to the map
                folium.PolyLine(
                    locations=coords,
                    color='red',
                    weight=4,
                    opacity=0.7,
                    popup=popup_text if popup_text else "Route"
                ).add_to(map_obj)
            else:
                # If no geometry, use node coordinates
                try:
                    u_coords = (road_graph.nodes[u]['y'], road_graph.nodes[u]['x'])
                    v_coords = (road_graph.nodes[v]['y'], road_graph.nodes[v]['x'])
                    
                    folium.PolyLine(
                        locations=[u_coords, v_coords],
                        color='red',
                        weight=4,
                        opacity=0.7,
                        popup=popup_text if popup_text else "Route"
                    ).add_to(map_obj)
                except KeyError:
                    print(f"Error: Missing coordinate data for nodes {u} or {v}")
        
        # Add markers for start and end
        try:
            # Start marker
            start_coords = (road_graph.nodes[path_nodes[0]]['y'], road_graph.nodes[path_nodes[0]]['x'])
            folium.Marker(
                location=start_coords,
                popup="Start",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(map_obj)
            
            # End marker
            end_coords = (road_graph.nodes[path_nodes[-1]]['y'], road_graph.nodes[path_nodes[-1]]['x'])
            folium.Marker(
                location=end_coords,
                popup="End",
                icon=folium.Icon(color='red', icon='stop', prefix='fa')
            ).add_to(map_obj)
        except (KeyError, IndexError) as e:
            print(f"Error adding start/end markers: {e}")
    
    except Exception as e:
        print(f"Error plotting route on map: {e}")
        traceback.print_exc()