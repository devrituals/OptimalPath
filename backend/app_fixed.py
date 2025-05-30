import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.features import DivIcon
import streamlit_folium
from streamlit_js_eval import get_geolocation
import time
import traceback
from math import radians, sin, cos, sqrt, atan2
import pickle
from pathlib import Path

# Configure OSMnx settings
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = ".cache"

# Helper functions
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

def get_road_network(place_name="Marrakech, Morocco", network_type="drive", force_refresh=False):
    """Get a detailed road network with proper road geometries"""
    # Create cache directory if needed
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache filename
    safe_name = place_name.replace(' ', '_').replace(',', '')
    cache_file = cache_dir / f"{safe_name}_{network_type}_detailed.pkl"
    
    # Check if cache exists and not forcing refresh
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, 'rb') as f:
                G = pickle.load(f)
            st.success(f"Loaded road network from cache")
            return G
        except Exception as e:
            st.warning(f"Error loading cached network: {e}")
    
    # Create new network
    try:
        st.info(f"Downloading detailed road network for {place_name}...")
        
        # Try place name approach with simplified=False to keep road geometries
        try:
            # Get a more detailed network with all roads
            G = ox.graph_from_place(place_name, network_type=network_type, simplified=False)
        except Exception as e:
            st.warning(f"Place-based network download failed: {e}")
            
            # Fall back to bounding box for Marrakech
            st.info("Falling back to bounding box...")
            north, south, east, west = 31.68, 31.57, -7.93, -8.04
            G = ox.graph_from_bbox(north, south, east, west, network_type=network_type, simplified=False)
        
        # Add edge speeds and travel times
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        
        # Project the graph to use consistent distance units
        G = ox.project_graph(G)

        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
            st.success("Network saved to cache")
        except Exception as e:
            st.warning(f"Could not save to cache: {e}")
        
        return G
    except Exception as e:
        st.error(f"Error getting road network: {e}")
        if st.session_state.debug_mode:
            st.error(traceback.format_exc())
        return create_fallback_network()


def create_fallback_network():
    """Create a minimal fallback network of Marrakech landmarks"""
    st.warning("Creating fallback network with key landmarks")
    G = nx.Graph()
    
    # Key locations in Marrakech
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
    for loc, (lat, lon) in locations.items():
        G.add_node(loc, y=lat, x=lon)
    
    # Add edges between all pairs
    nodes = list(G.nodes())
    for i, n1 in enumerate(nodes):
        lat1, lon1 = G.nodes[n1]['y'], G.nodes[n1]['x']
        for n2 in nodes[i+1:]:
            lat2, lon2 = G.nodes[n2]['y'], G.nodes[n2]['x']
            
            # Calculate distance
            dist = calculate_haversine_distance(lat1, lon1, lat2, lon2)
            time_min = dist * 12  # Approx 5 km/h walking speed
            
            # Add bidirectional edges
            G.add_edge(n1, n2, length=dist, time=time_min)
            G.add_edge(n2, n1, length=dist, time=time_min)
    
    return G

def find_shortest_path_exact(G, start_coords, end_coords, weight='length'):
    """
    Find the most accurate road-following path between two points using an improved method
    that tries multiple nearby nodes to find the truly shortest route.

    
    Args:
        G: NetworkX graph representing the road network
        start_coords: (lat, lon) tuple for start point
        end_coords: (lat, lon) tuple for end point
        weight: 'length' or 'travel_time'
        
    Returns:
        route: List of node IDs in the path
        distance: Total distance/time of the path in appropriate units
    """
    # Find multiple nearby nodes to try - this improves accuracy
    try:
        start_nodes = ox.distance.nearest_nodes(G, X=[start_coords[1]], Y=[start_coords[0]], n=5)
        end_nodes = ox.distance.nearest_nodes(G, X=[end_coords[1]], Y=[end_coords[0]], n=5)
        
        # Make sure we have lists
        if not isinstance(start_nodes, list):
            start_nodes = [start_nodes]
        if not isinstance(end_nodes, list):
            end_nodes = [end_nodes]
    except:
        # Fallback to single nearest node
        try:
            start_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
            end_node = ox.distance.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])
            start_nodes = [start_node]
            end_nodes = [end_node]
        except:
            # Last resort manual calculation
            start_node = None
            min_dist = float('inf')
            for node, data in G.nodes(data=True):
                try:
                    node_y = data.get('y')
                    node_x = data.get('x')
                    if node_y and node_x:
                        dist = ((start_coords[0] - node_y)**2 + (start_coords[1] - node_x)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            start_node = node
                except:
                    continue
            
            end_node = None
            min_dist = float('inf')
            for node, data in G.nodes(data=True):
                try:
                    node_y = data.get('y')
                    node_x = data.get('x')
                    if node_y and node_x:
                        dist = ((end_coords[0] - node_y)**2 + (end_coords[1] - node_x)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            end_node = node
                except:
                    continue
                    
            if start_node is None or end_node is None:
                return None, calculate_haversine_distance(
                    start_coords[0], start_coords[1], 
                    end_coords[0], end_coords[1]
                )
                
            start_nodes = [start_node]
            end_nodes = [end_node]
    
    # Try all combinations to find the truly shortest route
    best_route = None
    best_length = float('inf')
    
    for s_node in start_nodes:
        for e_node in end_nodes:
            try:
                route = nx.shortest_path(G, s_node, e_node, weight=weight)
                
                # Calculate route length
                if weight == 'length':
                    length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
                    # Convert to km
                    length = length / 1000
                else:  # travel_time
                    length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'travel_time'))
                    # Convert to minutes
                    length = length / 60

                
                # Keep the best route
                if length < best_length:
                    best_route = route
                    best_length = length
            except:
                continue
    
    # If we found a route, return it
    if best_route:
        return best_route, best_length
    
    # Fallback to direct distance
    direct_dist = calculate_haversine_distance(
        start_coords[0], start_coords[1],
        end_coords[0], end_coords[1]
    )
    
    # Convert to minutes if needed
    if weight == 'travel_time':
        direct_dist = direct_dist * 12  # ~5 km/h walking speed
        
    return None, direct_dist

def get_marrakech_pedestrian_network():
    """Get a pedestrian-focused network for Marrakech that follows actual walking paths"""
    cache_file = Path(".cache") / "marrakech_pedestrian.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    # Use WALK network type - critical for Marrakech's narrow streets
    try:
        # North, south, east, west boundaries of Marrakech
        north, south, east, west = 31.68, 31.57, -7.93, -8.04
        G = ox.graph_from_bbox(north, south, east, west, network_type="walk", simplify=False)
        
        # Save to cache for future use
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(G, f)
        
        return G
    except Exception as e:
        st.error(f"Error getting pedestrian network: {e}")
        return create_fallback_network()
    
def plot_route_with_osmnx(start_loc, end_loc, locations_dict):
    """Create a pedestrian route map that follows streets properly"""
    # Get coordinates
    start_coords = locations_dict[start_loc]  # (lat, lon)
    end_coords = locations_dict[end_loc]  # (lat, lon)
    
    # Get pedestrian network
    G = get_marrakech_pedestrian_network()
    
    # Find nearest nodes - note longitude (X) first, latitude (Y) second
    origin_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
    destination_node = ox.distance.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])
    
    # Find shortest route (use length weight for pedestrians)
    route = nx.shortest_path(G, origin_node, destination_node, weight="length")
    
    # Calculate distance
    route_length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length')) / 1000
    
    # Create map
    route_map = folium.Map(location=start_coords, zoom_start=15, tiles="OpenStreetMap")
    
    # Get exact coordinates for route plotting
    route_coords = []
    route_coords.append(start_coords)  # Start exactly at marker
    
    for node in route:
        y = G.nodes[node]['y']
        x = G.nodes[node]['x']
        route_coords.append((y, x))
    
    route_coords.append(end_coords)  # End exactly at marker
    
    # Draw route as a single polyline to avoid doubled lines
    folium.PolyLine(
        locations=route_coords,
        color='blue',
        weight=4,
        opacity=0.8
    ).add_to(route_map)
    
    # Add markers
    folium.Marker(
        location=start_coords,
        popup=f"Start: {start_loc}",
        icon=folium.Icon(color='green')
    ).add_to(route_map)
    
    folium.Marker(
        location=end_coords,
        popup=f"End: {end_loc}",
        icon=folium.Icon(color='red')
    ).add_to(route_map)
    
    # Add distance label
    folium.Marker(
        location=[(start_coords[0] + end_coords[0])/2, (start_coords[1] + end_coords[1])/2],
        icon=folium.DivIcon(
            icon_size=(120, 30),
            icon_anchor=(60, 15),
            html=f'<div style="background-color: white; padding: 3px 8px; border-radius: 5px; text-align: center; box-shadow: 0 0 3px rgba(0,0,0,0.3);"><b>{route_length:.2f} km</b></div>'
        )
    ).add_to(route_map)
    
    return route_map, route_length

def plot_route_on_folium_map(G, route, m=None, start_coords=None, end_coords=None, route_color='blue', weight=4):
    """Plot a route on a folium map using OSMnx's plot_route_folium"""
    # Create a map if one wasn't provided
    if m is None:
        m = folium.Map(location=start_coords, zoom_start=14)
    
    # Use OSMnx's built-in function to plot the route on the map
    if route is not None:
        try:
            # This will respect actual road geometries
            m = ox.plot_route_folium(G, route, route_map=m, color=route_color, weight=weight)
        except Exception as e:
            st.warning(f"Error plotting detailed route: {e}")
            # Fallback to simpler line plotting if necessary
            try:
                # Extract node coordinates
                route_lats = []
                route_lons = []
                for node in route:
                    try:
                        route_lats.append(G.nodes[node]['y'])
                        route_lons.append(G.nodes[node]['x'])
                    except:
                        continue
                
                # Create coordinates list with start and end included
                if start_coords:
                    route_coords = [start_coords]
                else:
                    route_coords = []
                    
                for i in range(len(route_lats)):
                    route_coords.append((route_lats[i], route_lons[i]))
                    
                if end_coords:
                    route_coords.append(end_coords)
                
                # Plot as simple line
                folium.PolyLine(
                    locations=route_coords,
                    color=route_color,
                    weight=weight,
                    opacity=0.7
                ).add_to(m)
            except Exception as e:
                st.error(f"Fallback route plotting also failed: {e}")
    
    return m

def create_simplified_map(center_coords, markers):
    """Create a simplified map for fallback"""
    m = folium.Map(location=center_coords, zoom_start=13)
    
    for lat, lon, name in markers:
        try:
            folium.Marker(location=[lat, lon], popup=name).add_to(m)
        except:
            pass
            
    return m

# Marrakech sample graph for location list
def create_marrakech_sample_graph():
    """Create a simple graph with key Marrakech locations"""
    G = nx.Graph()
    
    # Add key locations
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
    for loc, coords in locations.items():
        G.add_node(loc, pos=coords)
    
    return G

def get_optimized_graph(area_name="Marrakech", cache=True):
    """Get an optimized walking graph for Marrakech medina areas"""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{area_name}_pedestrian_detailed.pkl"
    
    if cache_file.exists() and cache:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Use network_type="walk" for Marrakech medina areas - critical for accuracy
    try:
        # Focus specifically on the popular tourist areas
        G = ox.graph_from_place("Medina, Marrakech, Morocco", network_type="walk", simplify=False)
    except:
        # Fallback - exact coords of Marrakech medina
        north, south, east, west = 31.64, 31.61, -7.97, -8.01
        G = ox.graph_from_bbox(north, south, east, west, network_type="walk", simplify=False)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(G, f)
    
    return G

def find_truly_shortest_path(G, start_coords, end_coords):
    """Find the actual shortest path between two points"""
    # Find nearest nodes - note longitude first, latitude second
    start_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
    end_node = ox.distance.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])
    
    # Try different routing weights to find the truly shortest path
    try:
        # Most accurate for Marrakech medina: use length first
        return nx.shortest_path(G, start_node, end_node, weight='length')
    except:
        # Try with no weight specification as a fallback
        try:
            return nx.shortest_path(G, start_node, end_node)
        except:
            return None

def plot_optimized_route(G, start_coords, end_coords, m=None):
    """Plot a route that actually follows streets and starts at markers"""
    if m is None:
        m = folium.Map(location=start_coords, zoom_start=16, tiles="OpenStreetMap")
    
    # Find shortest path that follows streets
    route = find_truly_shortest_path(G, start_coords, end_coords)
    
    if route:
        # Create coordinates list that starts and ends at exact markers
        route_coords = [start_coords]  # Start at exact marker
        
        # Add nodes along the route
        for node in route:
            try:
                y, x = G.nodes[node]['y'], G.nodes[node]['x']
                route_coords.append((y, x))
            except:
                pass
                
        route_coords.append(end_coords)  # End at exact marker
        
        # Draw the route
        folium.PolyLine(
            locations=route_coords,
            color='blue',
            weight=4,
            opacity=0.8
        ).add_to(m)
        
        # Calculate length
        length = 0
        for i in range(len(route_coords)-1):
            length += calculate_haversine_distance(
                route_coords[i][0], route_coords[i][1],
                route_coords[i+1][0], route_coords[i+1][1]
            )
        
        return m, length
    
    # Fallback if routing fails
    folium.PolyLine(
        locations=[start_coords, end_coords],
        color='red', 
        weight=3,
        opacity=0.7,
        dash_array='5'
    ).add_to(m)
    
    return m, calculate_haversine_distance(
        start_coords[0], start_coords[1],
        end_coords[0], end_coords[1]
    )

def convert_graph_to_coordinates(G):
    """Convert graph to location dictionary"""
    pos = nx.get_node_attributes(G, 'pos')
    return pos

def sort_locations_by_proximity(selected_locations, locations_dict, start_coords):
    """Sort locations by their distance from the starting point"""
    locations_with_distances = []
    
    for loc in selected_locations:
        loc_coords = locations_dict[loc]
        distance = calculate_haversine_distance(
            start_coords[0], start_coords[1],
            loc_coords[0], loc_coords[1]
        )
        locations_with_distances.append((loc, distance))
    
    # Sort by distance (closest first)
    locations_with_distances.sort(key=lambda x: x[1])
    
    # Return just the location names in order
    return [loc for loc, _ in locations_with_distances]

def plot_multi_stop_route(G, ordered_route, locations_dict, start_coords, weight='length'):
    """
    Create a map with the multi-stop route properly visualized.
    
    Args:
        G: Road network graph
        ordered_route: List of location names in optimal order
        locations_dict: Dictionary mapping location names to coordinates
        start_coords: Starting coordinates (lat, lon)
        weight: 'length' or 'travel_time'
        
    Returns:
        m: Folium map with the route
        total_distance: Total route distance
    """
    # Create a base map
    m = folium.Map(location=(31.6295, -7.9811), zoom_start=13)
    
    # Add title
    title_html = f'<h3 align="center" style="font-size:16px"><b>Multi-Stop Route</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Calculate total distance
    total_distance = 0
    
    # First segment: from start to first stop
    first_stop_coords = locations_dict[ordered_route[0]]
    
    # Find the path from start to first location
    start_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
    first_node = ox.distance.nearest_nodes(G, X=first_stop_coords[1], Y=first_stop_coords[0])
    
    try:
        route_to_first = nx.shortest_path(G, start_node, first_node, weight=weight)
        if weight == 'length':
            first_distance = sum(ox.utils_graph.get_route_edge_attributes(G, route_to_first, 'length')) / 1000
        else:
            first_distance = sum(ox.utils_graph.get_route_edge_attributes(G, route_to_first, 'travel_time')) / 60
        
        # Get coordinates for the first segment
        segment_coords = []
        segment_coords.append(start_coords)  # Exact start
        
        for node in route_to_first:
            y = G.nodes[node]['y']
            x = G.nodes[node]['x']
            segment_coords.append((y, x))
        
        segment_coords.append(first_stop_coords)  # Exact end
        
        # Plot the first segment
        folium.PolyLine(
            locations=segment_coords,
            color='blue',
            weight=5,
            opacity=0.8
        ).add_to(m)
        
        total_distance += first_distance
    except Exception as e:
        # Fallback to direct line
        folium.PolyLine(
            locations=[start_coords, first_stop_coords],
            color='blue',
            weight=4,
            opacity=0.8,
            dash_array='5'
        ).add_to(m)
        
        # Calculate direct distance
        direct_dist = calculate_haversine_distance(
            start_coords[0], start_coords[1],
            first_stop_coords[0], first_stop_coords[1]
        )
        total_distance += direct_dist
    
    # Connect all remaining stops in order
    for i in range(len(ordered_route)-1):
        from_loc = ordered_route[i]
        to_loc = ordered_route[i+1]
        
        from_coords = locations_dict[from_loc]
        to_coords = locations_dict[to_loc]
        
        # Find path between stops
        from_node = ox.distance.nearest_nodes(G, X=from_coords[1], Y=from_coords[0])
        to_node = ox.distance.nearest_nodes(G, X=to_coords[1], Y=to_coords[0])
        
        try:
            segment_route = nx.shortest_path(G, from_node, to_node, weight=weight)
            if weight == 'length':
                segment_distance = sum(ox.utils_graph.get_route_edge_attributes(G, segment_route, 'length')) / 1000
            else:
                segment_distance = sum(ox.utils_graph.get_route_edge_attributes(G, segment_route, 'travel_time')) / 60
            
            # Get coordinates for this segment
            segment_coords = []
            segment_coords.append(from_coords)  # Exact start
            
            for node in segment_route:
                y = G.nodes[node]['y']
                x = G.nodes[node]['x']
                segment_coords.append((y, x))
            
            segment_coords.append(to_coords)  # Exact end
            
            # Plot this segment
            folium.PolyLine(
                locations=segment_coords,
                color='red',
                weight=5,
                opacity=0.8
            ).add_to(m)
            
            total_distance += segment_distance
            
            # Add distance label at midpoint
            mid_idx = len(segment_coords) // 2
            if mid_idx > 0:
                mid_point = segment_coords[mid_idx]
                folium.Marker(
                    location=mid_point,
                    icon=folium.DivIcon(
                        icon_size=(80, 20),
                        icon_anchor=(40, 10),
                        html=f'<div style="font-size: 10pt; color: red; background-color: white; padding: 2px 5px; border-radius: 3px; text-align: center;">{segment_distance:.1f} {("km" if weight=="length" else "min")}</div>'
                    )
                ).add_to(m)
        except Exception as e:
            # Fallback to direct line
            folium.PolyLine(
                locations=[from_coords, to_coords],
                color='red',
                weight=4,
                opacity=0.8,
                dash_array='5'
            ).add_to(m)
            
            # Calculate direct distance
            direct_dist = calculate_haversine_distance(
                from_coords[0], from_coords[1],
                to_coords[0], to_coords[1]
            )
            total_distance += direct_dist
    
    # Add start marker
    folium.Marker(
        location=start_coords,
        popup="Start",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add numbered markers for each stop
    for i, loc in enumerate(ordered_route):
        coords = locations_dict[loc]
        
        # Add circle marker with number
        folium.CircleMarker(
            location=coords,
            radius=18,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.9,
            popup=f"Stop {i+1}: {loc}"
        ).add_to(m)
        
        # Add number
        folium.Marker(
            location=coords,
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>'
            )
        ).add_to(m)
    
    return m, total_distance

def find_multi_stop_optimized(G, locations, start_coords):
    """Find optimized multi-stop route starting from nearest location"""
    # 1. First, explicitly sort locations by distance from start
    locations_with_distances = []
    for loc_name, loc_coords in locations.items():
        dist = calculate_haversine_distance(
            start_coords[0], start_coords[1], 
            loc_coords[0], loc_coords[1]
        )
        locations_with_distances.append((loc_name, loc_coords, dist))
    
    # 2. Sort by distance - nearest first
    locations_with_distances.sort(key=lambda x: x[2])
    
    # 3. Get ordered location names
    ordered_locations = [loc[0] for loc in locations_with_distances]
    
    return ordered_locations

def find_multi_stop_route(G, selected_locations, locations_dict, start_coords, weight='length'):
    """
    Find the best multi-stop route that starts from the closest location
    to the delivery person and visits all selected locations.
    
    Args:
        G: Road network graph
        selected_locations: List of location names to visit
        locations_dict: Dictionary mapping location names to coordinates
        start_coords: Starting coordinates (lat, lon)
        weight: 'length' or 'travel_time'
        
    Returns:
        ordered_route: List of location names in optimal order
        first_stop: Name of the first stop
        total_distance: Total route distance
    """
    # Find which location is closest to the starting point
    min_distance = float('inf')
    closest_location = None
    closest_index = 0
    
    for i, loc in enumerate(selected_locations):
        loc_coords = locations_dict[loc]
        
        # Calculate direct distance first
        direct_dist = calculate_haversine_distance(
            start_coords[0], start_coords[1],
            loc_coords[0], loc_coords[1]
        )
        
        if direct_dist < min_distance:
            min_distance = direct_dist
            closest_location = loc
            closest_index = i
    
        sorted_locations = sort_locations_by_proximity(selected_locations, locations_dict, start_coords)
        selected_locations = sorted_locations
        
    # Create distance matrix between all locations for TSP
    n = len(selected_locations)
    distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Show progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Calculating distances between locations...")
    
    total_calculations = n * n
    completed = 0
    
    # Calculate all pairwise distances
    for i in range(n):
        for j in range(n):
            # Update progress
            completed += 1
            progress = completed / total_calculations
            progress_bar.progress(progress)
            
            if i == j:
                distance_matrix[i][j] = 0.0
                continue
            
            from_coords = locations_dict[selected_locations[i]]
            to_coords = locations_dict[selected_locations[j]]
            
            # Find nearest nodes
            try:
                from_node = ox.distance.nearest_nodes(G, X=from_coords[1], Y=from_coords[0])
                to_node = ox.distance.nearest_nodes(G, X=to_coords[1], Y=to_coords[0])
            
                # Calculate shortest path
                try:
                    route = nx.shortest_path(G, from_node, to_node, weight=weight)
                    if weight == 'length':
                        distance = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length')) / 1000  # to km
                    else:
                        distance = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'travel_time')) / 60  # to min
                    
                    distance_matrix[i][j] = max(distance, 0.001)
                except:
                    # Fallback to direct distance
                    direct_dist = calculate_haversine_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
                    distance_matrix[i][j] = max(direct_dist, 0.001)
            except:
                # Fallback if node finding fails
                direct_dist = calculate_haversine_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
                distance_matrix[i][j] = max(direct_dist, 0.001)
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()

    min_distance = float('inf')
    closest_location_index = 0

    for i, loc in enumerate(selected_locations):
        loc_coords = locations_dict[loc]
        direct_dist = calculate_haversine_distance(
            start_coords[0], start_coords[1],
            loc_coords[0], loc_coords[1]
        )
        
        if direct_dist < min_distance:
            min_distance = direct_dist
            closest_location_index = i


    # Solve TSP with closest location as the starting point
    try:
        from optimal_path import solve_tsp
        route_indices = solve_tsp(distance_matrix, start_index=closest_index, locations=selected_locations)
    except ImportError:
        # Fallback TSP solver
        route_indices = nearest_neighbor_tsp(distance_matrix, start_index=closest_index)
    
    # Calculate total route distance
    total_distance = 0.0
    for i in range(len(route_indices)-1):
        idx1 = route_indices[i]
        idx2 = route_indices[i+1]
        total_distance += distance_matrix[idx1][idx2]
    
    # Create ordered route list
    ordered_route = [selected_locations[i] for i in route_indices]
    
    return ordered_route, closest_location, total_distance

def nearest_neighbor_tsp(distance_matrix, start_index=0):
    """Simple nearest neighbor TSP solver"""
    n = len(distance_matrix)
    path = [start_index]
    unvisited = list(range(n))
    unvisited.remove(start_index)
    
    current = start_index
    while unvisited:
        next_idx = min(unvisited, key=lambda x: distance_matrix[current][x])
        path.append(next_idx)
        unvisited.remove(next_idx)
        current = next_idx
    
    return path


def create_distance_matrix_improved(G, locations_dict, location_names, weight='length'):
    """
    Create a more accurate distance matrix for TSP calculations
    """
    n = len(location_names)
    matrix = []
    
    # Show progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Calculating distances between locations...")
    
    # Calculate distances between all pairs
    total_calculations = n * n
    completed = 0
    
    for i in range(n):
        row = []
        for j in range(n):
            # Update progress
            completed += 1
            progress = completed / total_calculations
            progress_bar.progress(progress)
            
            if i == j:
                # Zero distance for same location
                row.append(0)
                continue
            
            # Get coordinates
            start_coords = locations_dict[location_names[i]]
            end_coords = locations_dict[location_names[j]]
            
            # Find shortest path with improved accuracy
            _, distance = find_shortest_path_exact(G, start_coords, end_coords, weight)
            
            # Store in matrix with minimum value
            row.append(max(distance, 0.001))
        
        matrix.append(row)
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    return matrix

# Import the TSP solver from your existing module or use a simple one
try:
    from optimal_path import solve_tsp
except ImportError:
    # Fallback TSP solver if your module isn't available
    def solve_tsp(distance_matrix, start_index=0, delivery_person_location=None, locations=None):
        """Simple nearest neighbor TSP solver"""
        n = len(distance_matrix)
        path = [start_index]
        unvisited = list(range(n))
        unvisited.remove(start_index)
        
        # Find nearest neighbors
        current = start_index
        while unvisited:
            next_idx = min(unvisited, key=lambda x: distance_matrix[current][x])
            path.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx
            
        return path

# Main app function
def main():
    # App title and configuration
    st.set_page_config(page_title="Marrakech Route Planner", layout="wide")
    
    # App title
    st.title("Marrakech Interactive Route Planner")
    st.write("""
    This application helps you plan your routes in Marrakech. You can find the shortest path between two locations 
    or plan an optimal route through multiple locations. The routes follow the actual road network.
    """)
    
    # Default Marrakech center
    DEFAULT_MARRAKECH_COORDS = (31.6295, -7.9811)
    
    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'location_detected' not in st.session_state:
        st.session_state.location_detected = False
    if 'delivery_person_coords' not in st.session_state:
        st.session_state.delivery_person_coords = None
    if 'geolocation_error' not in st.session_state:
        st.session_state.geolocation_error = None
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Shortest Path", "Multi-Stop Route"])
    
    # Create sample graph for locations
    sample_G = create_marrakech_sample_graph()
    locations = list(sample_G.nodes())
    locations_dict = convert_graph_to_coordinates(sample_G)
    
    # Shortest Path tab
    with tab1:
        st.header("Find the Shortest Path")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_loc = st.selectbox("Select starting location", locations, index=0)
            end_loc = st.selectbox("Select destination", locations, index=3)
            cost_type = st.radio("Optimize by", ["distance", "time"], horizontal=True)
        
        if st.button("Find Shortest Path"):
            with st.spinner("Finding the shortest path..."):
                try:
                    # Use the direct function that follows roads properly
                    m, distance = plot_route_with_osmnx(start_loc, end_loc, locations_dict)
                    
                    # Display results
                    with col2:
                        if cost_type == 'distance':
                            st.success(f"Path found! Total distance: {distance:.2f} km")
                        else:
                            # Convert distance to time (rough estimate)
                            time_minutes = distance * 12  # Assuming 5 km/h walking speed
                            st.success(f"Path found! Estimated time: {time_minutes:.1f} minutes")
                        
                        st.write("Route:")
                        st.write(f"1. {start_loc}")
                        st.write(f"2. {end_loc}")
                    
                    # Display map
                    st.subheader("Interactive Map")
                    streamlit_folium.folium_static(m, width=800, height=500)
                except Exception as e:
                    st.error(f"Error with direct routing method: {e}")
                    if st.session_state.debug_mode:
                        st.error(traceback.format_exc())
                    
                    # Fall back to alternative method
                    try:
                        # Get coordinates
                        start_coords = locations_dict[start_loc]
                        end_coords = locations_dict[end_loc]
                        
                        # Get the road network
                        with st.status("Loading road network..."):
                            G = get_road_network("Marrakech, Morocco", "drive")
                        
                        # Find route
                        with st.status("Finding route..."):
                            weight = 'length' if cost_type == 'distance' else 'travel_time'
                            route, distance = find_shortest_path_exact(G, start_coords, end_coords, weight)
                        
                        # Display results
                        with col2:
                            unit = 'km' if cost_type == 'distance' else 'minutes'
                            st.success(f"Path found! Total {cost_type}: {distance:.2f} {unit}")
                            st.write("Route:")
                            st.write(f"1. {start_loc}")
                            st.write(f"2. {end_loc}")
                        
                        # Create the map
                        with st.status("Creating map..."):
                            # Create folium map
                            m = folium.Map(location=DEFAULT_MARRAKECH_COORDS, zoom_start=13)
                            
                            # Add title
                            title_html = f'<h3 align="center" style="font-size:16px"><b>Route from {start_loc} to {end_loc}</b></h3>'
                            m.get_root().html.add_child(folium.Element(title_html))
                            
                            # Plot route using OSMnx's plot_route_folium
                            m = plot_route_on_folium_map(
                                G, 
                                route, 
                                m=m,
                                start_coords=start_coords,
                                end_coords=end_coords,
                                route_color='blue',
                                weight=5
                            )
                            
                            # Add markers for start and end
                            folium.Marker(
                                location=start_coords,
                                popup=f"Start: {start_loc}",
                                tooltip=f"Start: {start_loc}",
                                icon=folium.Icon(color='green', icon='play', prefix='fa')
                            ).add_to(m)
                            
                            folium.Marker(
                                location=end_coords,
                                popup=f"End: {end_loc}",
                                tooltip=f"End: {end_loc}",
                                icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
                            ).add_to(m)
                            
                            # Add distance display
                            midpoint = (
                                (start_coords[0] + end_coords[0]) / 2,
                                (start_coords[1] + end_coords[1]) / 2
                            )
                            folium.Marker(
                                location=midpoint,
                                icon=folium.DivIcon(
                                    icon_size=(150, 36),
                                    icon_anchor=(75, 18),
                                    html=f'<div style="font-size: 12pt; background-color: white; padding: 5px 10px; border-radius: 5px; border: 1px solid blue; text-align: center; box-shadow: 2px 2px 4px #888;"><b>{distance:.2f} {unit}</b></div>'
                                )
                            ).add_to(m)
                            
                            # Display map
                            st.subheader("Interactive Map")
                            streamlit_folium.folium_static(m, width=800, height=500)
                    except Exception as e:
                        st.error(f"Error with fallback routing method: {e}")
                        if st.session_state.debug_mode:
                            st.error(traceback.format_exc())
                        
                        # Fall back to simplified map
                        markers = [
                            (start_coords[0], start_coords[1], f"Start: {start_loc}"),
                            (end_coords[0], end_coords[1], f"End: {end_loc}")
                        ]
                        m = create_simplified_map(DEFAULT_MARRAKECH_COORDS, markers)
                        st.subheader("Simplified Map (Fallback)")
                        streamlit_folium.folium_static(m, width=800, height=500)
    
    # Multi-Stop Route tab
    with tab2:
        st.header("Plan a Multi-Stop Route")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Select locations to visit:")
            selected_locations = []
            
            # Location selection
            for loc in locations:
                # Set default locations
                is_default = (loc == "Jemaa el-Fnaa" or loc == "Medina" or loc == "Gueliz")
                if st.checkbox(loc, value=is_default, key=f"loc_{loc}"):
                    selected_locations.append(loc)
            
            # Show number of selected locations
            st.write(f"Selected locations: {len(selected_locations)}")
            if len(selected_locations) < 2:
                st.warning("Please select at least two locations to visit.")
            
            # Start location selection
            location_method = st.radio(
                "Choose your starting location:",
                ["Use a named location", "Use my current location"],
                index=0
            )
            
            if location_method == "Use a named location":
                # Manual selection
                delivery_person_location = st.selectbox("Select starting location:", locations, index=0)
                start_coords = locations_dict[delivery_person_location]
            else:
                # Geolocation detection
                st.write("Click the button below to detect your current location:")
                
                if st.button("Detect Location"):
                    with st.spinner("Detecting your location..."):
                        try:
                            location = get_geolocation()
                            
                            if location and 'coords' in location:
                                lat = location['coords']['latitude']
                                lon = location['coords']['longitude']
                                
                                if -90 <= lat <= 90 and -180 <= lon <= 180:
                                    st.success(f"📍 Location detected: ({lat:.5f}, {lon:.5f})")
                                    start_coords = (lat, lon)
                                    delivery_person_location = "Current Location"
                                else:
                                    st.error("Invalid coordinates received")
                                    # Fallback to default
                                    delivery_person_location = "Jemaa el-Fnaa"
                                    start_coords = locations_dict[delivery_person_location]
                            else:
                                st.error("Could not access your location")
                                # Fallback to default
                                delivery_person_location = "Jemaa el-Fnaa"
                                start_coords = locations_dict[delivery_person_location]
                        except Exception as e:
                            st.error(f"Error detecting location: {e}")
                            # Fallback to default
                            delivery_person_location = "Jemaa el-Fnaa"
                            start_coords = locations_dict[delivery_person_location]
                else:
                    # Default until detection
                    delivery_person_location = "Jemaa el-Fnaa"
                    start_coords = locations_dict[delivery_person_location]
            
            cost_type = st.radio("Optimize by", ["distance", "time"], horizontal=True, key="tsp_cost")
        
        # Find optimal route button
        if st.button("Find Optimal Route"):
            # Check minimum locations
            if len(selected_locations) < 2:
                st.error("Please select at least 2 locations to visit.")
            else:
                with st.spinner("Finding optimal route..."):
                    with col2:
                        st.write(f"Finding optimal route through {len(selected_locations)} locations...")
                        
                        # Get the road network
                        with st.status("Loading road network..."):
                            G = get_road_network("Marrakech, Morocco", "drive")
                        
                        # Create distance matrix
                        with st.status("Calculating distances between all locations..."):
                            weight = 'length' if cost_type == 'distance' else 'travel_time'
                            
                            # Create distance matrix using the road network
                            distance_matrix = create_distance_matrix_improved(
                                G, locations_dict, selected_locations, weight
                            )
                        
                        # Solve TSP
                        with st.status("Finding the optimal route order..."):
                            route_indices = solve_tsp(distance_matrix, locations=selected_locations)
                        
                        if route_indices:
                            route = [selected_locations[i] for i in route_indices]
                            
                            # Calculate total cost
                            total_cost = 0
                            for i in range(len(route_indices)-1):
                                idx1, idx2 = route_indices[i], route_indices[i+1]
                                total_cost += distance_matrix[idx1][idx2]
                            
                            # Calculate path from start to first stop
                            first_stop_coords = locations_dict[route[0]]
                            route_to_first, first_stop_cost = find_shortest_path_exact(
                                G, start_coords, first_stop_coords, weight
                            )
                            
                            # Format and display results
                            unit = 'km' if cost_type == 'distance' else 'minutes'
                            st.success(f"✅ Optimal route found! Total {cost_type}: {total_cost:.2f} {unit}")
                            st.write(f"Distance to first stop: {first_stop_cost:.2f} {unit}")
                            
                            st.write("Route:")
                            for i, loc in enumerate(route):
                                st.write(f"{i+1}. {loc}")

                # Create the map
                try:
                    with st.status("Creating interactive map..."):
                        # Create folium map
                        m = folium.Map(location=DEFAULT_MARRAKECH_COORDS, zoom_start=13)
                        
                        # Add title
                        title_html = f'<h3 align="center" style="font-size:16px"><b>Multi-Stop Route</b></h3>'
                        m.get_root().html.add_child(folium.Element(title_html))
                        
                        # Get road network for just this area to ensure better routing
                        bounds_coords = [start_coords] + [locations_dict[loc] for loc in route]
                        min_lat = min(coord[0] for coord in bounds_coords) - 0.03
                        max_lat = max(coord[0] for coord in bounds_coords) + 0.03
                        min_lon = min(coord[1] for coord in bounds_coords) - 0.03
                        max_lon = max(coord[1] for coord in bounds_coords) + 0.03
                        
                        # Create a dedicated network for this specific route area
                        route_G = ox.graph_from_bbox(max_lat, min_lat, max_lon, min_lon, network_type="drive")
                        
                        # Plot route from start location to first stop
                        first_stop_coords = locations_dict[route[0]]
                        
                        # Find nearest nodes - CRITICAL: note the longitude, latitude order
                        origin_node = ox.distance.nearest_nodes(route_G, X=start_coords[1], Y=start_coords[0])
                        first_dest_node = ox.distance.nearest_nodes(route_G, X=first_stop_coords[1], Y=first_stop_coords[0])
                        
                        # Find first route
                        route_to_first = nx.shortest_path(route_G, origin_node, first_dest_node, weight='length')
                        
                        # Get coordinates for the first route including exact start point
                        first_route_coords = []
                        first_route_coords.append(start_coords)  # Start exactly at marker
                        
                        for node in route_to_first:
                            y = route_G.nodes[node]['y']
                            x = route_G.nodes[node]['x']
                            first_route_coords.append((y, x))
                            
                        first_route_coords.append(first_stop_coords)  # End exactly at marker
                        
                        # Plot first route with custom PolyLine to ensure it connects to markers
                        folium.PolyLine(
                            locations=first_route_coords,
                            color='blue',
                            weight=5,
                            opacity=0.8
                        ).add_to(m)
                        
                        # Add routes between stops
                        for i in range(len(route)-1):
                            from_loc = route[i]
                            to_loc = route[i+1]
                            
                            from_coords = locations_dict[from_loc]
                            to_coords = locations_dict[to_loc]
                            
                            # Find nearest nodes
                            from_node = ox.distance.nearest_nodes(route_G, X=from_coords[1], Y=from_coords[0])
                            to_node = ox.distance.nearest_nodes(route_G, X=to_coords[1], Y=to_coords[0])
                            
                            # Calculate route
                            segment_route = nx.shortest_path(route_G, from_node, to_node, weight='length')
                            
                            # Get coordinates for this segment with exact start/end points
                            segment_coords = []
                            segment_coords.append(from_coords)  # Start exactly at marker
                            
                            for node in segment_route:
                                y = route_G.nodes[node]['y']
                                x = route_G.nodes[node]['x']
                                segment_coords.append((y, x))
                                
                            segment_coords.append(to_coords)  # End exactly at marker
                            
                            # Plot segment with custom PolyLine
                            folium.PolyLine(
                                locations=segment_coords,
                                color='red',
                                weight=5,
                                opacity=0.8
                            ).add_to(m)
                        
                        # Add start marker
                        folium.Marker(
                            location=start_coords,
                            popup=f"Start: {delivery_person_location}",
                            tooltip=f"Start: {delivery_person_location}",
                            icon=folium.Icon(color='green', icon='play', prefix='fa')
                        ).add_to(m)
                        
                        # Add numbered markers for each stop
                        for i, loc in enumerate(route):
                            coords = locations_dict[loc]
                            
                            # Circle with number
                            folium.CircleMarker(
                                location=coords,
                                radius=18,
                                color='red',
                                fill=True,
                                fill_color='red',
                                fill_opacity=0.9,
                                popup=f"Stop {i+1}: {loc}"
                            ).add_to(m)
                            
                            # Add number
                            folium.Marker(
                                location=coords,
                                icon=DivIcon(
                                    icon_size=(20, 20),
                                    icon_anchor=(10, 10),
                                    html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>'
                                )
                            ).add_to(m)
                        
                        # Display the map
                        st.subheader("Interactive Map")
                        streamlit_folium.folium_static(m, width=800, height=500)
                except Exception as e:
                    st.error(f"Error creating map: {e}")
                    if st.session_state.debug_mode:
                        st.error(traceback.format_exc())
    
    # Add footer
    st.markdown("---")
    st.markdown("### How to use this app")
    st.markdown("""
    1. **Shortest Path**: Select a starting point and destination to find the shortest route between them.
    2. **Multi-Stop Route**: Select multiple locations to visit, and the app will find the optimal order to visit them all.
       - You can either manually select a starting location or use your device's location.
       - Select at least 2 locations to visit.
       - The app uses road-based routing that follows actual street networks for realistic routes.
    
    The interactive map shows the route with colored lines and markers for each location.
    """)

if __name__ == "__main__":
    main()
