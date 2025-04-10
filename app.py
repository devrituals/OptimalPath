import streamlit as st
import folium
from folium.features import DivIcon
import networkx as nx
import streamlit_folium
from optimal_path import create_marrakech_sample_graph, find_shortest_path, solve_tsp, create_distance_matrix
from road_network import get_road_network, find_shortest_path_on_road, create_distance_matrix_on_road, convert_marrakech_graph_to_coordinates, get_nearest_node
from road_map import create_road_folium_map, plot_route_on_map

# Set page configuration
st.set_page_config(page_title="Marrakech Route Planner", layout="wide")

# Add title and description
st.title("Marrakech Interactive Route Planner")
st.write("""
This application helps you plan your routes in Marrakech. You can find the shortest path between two locations 
or plan an optimal route through multiple locations.
""")

# Create the graph
G = create_marrakech_sample_graph()

# Get all locations
locations = list(G.nodes())

# Convert locations to coordinates dictionary
locations_dict = convert_marrakech_graph_to_coordinates(G)

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Shortest Path", "Multi-Stop Route"])



# Function to create a Folium map with a path
def create_folium_map(graph, path=None, markers=True):
    # Get positions from node attributes
    pos = nx.get_node_attributes(graph, 'pos')
    
    # Find center of the map
    lats = [pos[node][0] for node in graph.nodes()]
    lons = [pos[node][1] for node in graph.nodes()]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create a map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    
    # Add all nodes as markers
    if markers:
        for node in graph.nodes():
            lat, lon = pos[node]
            color = 'blue'
            if path and node in path:
                color = 'red'
            
            folium.Marker(
                location=[lat, lon],
                popup=node,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
    
    # Add all edges
    for u, v, data in graph.edges(data=True):
        lat1, lon1 = pos[u]
        lat2, lon2 = pos[v]
        weight = data.get('distance', 1)
        
        # Check if this edge is part of the path
        edge_color = 'gray'
        edge_weight = 2
        edge_opacity = 0.5
        
        if path and len(path) > 1:
            path_edges = list(zip(path, path[1:]))
            if (u, v) in path_edges or (v, u) in path_edges:
                edge_color = 'red'
                edge_weight = 4
                edge_opacity = 1.0
        
        # Add the edge to the map
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color=edge_color,
            weight=edge_weight,
            opacity=edge_opacity,
            popup=f"{u} to {v}: {weight} km"
        ).add_to(m)
    
    # Add path information if provided
    if path and len(path) > 1:
        # Calculate total distance
        total_distance = 0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            # Check if there's a direct edge between u and v
            if v in graph[u] and 'distance' in graph[u][v]:
                total_distance += graph[u][v]['distance']
            else:
                # If no direct edge, find the shortest path between them
                temp_path, temp_cost = find_shortest_path(graph, u, v, 'distance')
                if temp_path:
                    total_distance += temp_cost
        
        # Add path information
        folium.Marker(
            location=[pos[path[0]][0], pos[path[0]][1]],
            icon=DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 12pt; color: red; font-weight: bold;">Start: {path[0]}</div>'
            )
        ).add_to(m)
        
        folium.Marker(
            location=[pos[path[-1]][0], pos[path[-1]][1]],
            icon=DivIcon(
                icon_size=(150,36),
                icon_anchor=(0,0),
                html=f'<div style="font-size: 12pt; color: red; font-weight: bold;">End: {path[-1]}</div>'
            )
        ).add_to(m)
    
    return m

# Shortest Path tab
with tab1:
    st.header("Find the Shortest Path")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_loc = st.selectbox("Select starting location", locations, index=0)
        end_loc = st.selectbox("Select destination", locations, index=3)
        cost_type = st.radio("Optimize by", ["distance", "time"], horizontal=True)
    
    if st.button("Find Shortest Path"):
        path, cost = find_shortest_path(G, start_loc, end_loc, cost_type)
        
        if path:
            with col2:
                st.success(f"Path found! Total {cost_type}: {cost} {'km' if cost_type == 'distance' else 'minutes'}")
                st.write("Route:")
                for i, loc in enumerate(path):
                    st.write(f"{i+1}. {loc}")
            
            # Create and display the map
            m = create_folium_map(G, path)
            st.subheader("Interactive Map")
            st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
        else:
            st.error(f"No path found between {start_loc} and {end_loc}")

# Multi-Stop Route tab
with tab2:
    st.header("Plan a Multi-Stop Route")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Select locations to visit:")
        selected_locations = []
        
        # Ensure at least one location is selected by default
        for loc in locations:
            # Set Jemaa el-Fnaa as default selected and make it more visible
            is_default = (loc == "Jemaa el-Fnaa")
            if st.checkbox(loc, value=is_default, key=f"loc_{loc}"):
                selected_locations.append(loc)
        
        # Debug information to help troubleshoot
        st.write(f"Selected locations: {len(selected_locations)}")
        if len(selected_locations) == 0:
            st.warning("Please select at least one location to visit.")
        
        st.write("Select delivery person's location:")
        delivery_person_location = st.selectbox("Delivery Person Location", locations, index=0)
        
        cost_type_tsp = st.radio("Optimize by", ["distance", "time"], horizontal=True, key="tsp_cost")
        
        routing_type = st.radio("Routing type", ["Graph-based", "Road-based"], horizontal=True, key="routing_type")
    
    if st.button("Find Optimal Route") and len(selected_locations) > 1:
        with col2:
            st.write(f"Finding optimal route through {len(selected_locations)} locations...")
            st.write(f"Starting from the nearest location to {delivery_person_location}")
            
            if routing_type == "Graph-based":
                # Create distance matrix using graph-based approach
                distance_matrix = create_distance_matrix(G, selected_locations, cost_type_tsp)
                
                # Solve TSP with delivery person location
                route_indices = solve_tsp(distance_matrix, delivery_person_location=delivery_person_location, locations=selected_locations)
                
                if route_indices:
                    route = [selected_locations[i] for i in route_indices]
                    
                    # Calculate total cost and collect path information
                    total_cost = 0
                    all_path_coords = []
                    all_markers = []
                    
                    # Add delivery person starting location
                    delivery_coords = locations_dict[delivery_person_location]
                    all_markers.append((delivery_coords[0], delivery_coords[1], f"Delivery Person: {delivery_person_location}"))
                    
                    # First, calculate path from delivery person to first stop
                    first_path, first_cost = find_shortest_path(G, delivery_person_location, route[0], cost_type_tsp)
                    if first_path:
                        total_cost += first_cost
                        path_coords = [(locations_dict[loc][0], locations_dict[loc][1]) for loc in first_path]
                        all_path_coords.extend(path_coords)
                    
                    # Process each location in the route
                    for i, location in enumerate(route):
                        # Add marker for current location
                        current_coords = locations_dict[location]
                        all_markers.append((current_coords[0], current_coords[1], f"Stop {i+1}: {location}"))
                        
                        # Calculate path to next location if not the last stop
                        if i < len(route) - 1:
                            path, cost = find_shortest_path(G, location, route[i+1], cost_type_tsp)
                            if path:
                                total_cost += cost
                                # Add path coordinates
                                path_coords = [(locations_dict[loc][0], locations_dict[loc][1]) for loc in path]
                                all_path_coords.extend(path_coords)
                    
                    # Remove duplicate coordinates to smooth the path
                    all_path_coords = [coord for i, coord in enumerate(all_path_coords)
                                      if i == 0 or coord != all_path_coords[i-1]]
                    # Create and display the map with the complete route
                    m = create_folium_map(G, all_path_coords)
                    st.subheader("Interactive Map")
                    st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
                    
                    # Display route information
                    st.success(f"Route found! Total {cost_type_tsp}: {total_cost} {'km' if cost_type_tsp == 'distance' else 'minutes'}")
                    st.write("Route order:")
                    st.write(f"1. Start at {delivery_person_location}")
                    for i, loc in enumerate(route):
                        st.write(f"{i+2}. {loc}")
                    
                    # Format distance with 3 decimal places if it's in kilometers
                    if cost_type_tsp == 'distance':
                        formatted_cost = f"{total_cost:.3f}"
                    else:
                        formatted_cost = f"{total_cost}"
                    st.success(f"Optimal route found! Total {cost_type_tsp}: {formatted_cost} {'km' if cost_type_tsp == 'distance' else 'minutes'}")
                    # Format first stop distance with 3 decimal places if it's in kilometers
                    if cost_type_tsp == 'distance':
                        formatted_first_stop_cost = f"{first_stop_cost:.3f}"
                    else:
                        formatted_first_stop_cost = f"{first_stop_cost}"
                    st.write(f"Distance from {delivery_person_location} to first stop: {formatted_first_stop_cost} {'km' if cost_type_tsp == 'distance' else 'minutes'}")
                    st.write("Route:")
                    for i, loc in enumerate(route):
                        st.write(f"{i+1}. {loc}")
                    
                    # Create a complete path that includes all intermediate nodes
                    complete_path = []
                    for i in range(len(route)-1):
                        segment_path, _ = find_shortest_path(G, route[i], route[i+1], cost_type_tsp)
                        if segment_path:
                            # Add all nodes except the last one (to avoid duplicates)
                            if i < len(route)-2:
                                complete_path.extend(segment_path[:-1])
                            else:
                                complete_path.extend(segment_path)
                    
                    # Create and display the map
                    m = create_folium_map(G, complete_path, markers=False)  # Don't add default markers
                    
                    # Add delivery person marker with a distinctive icon
                    pos = nx.get_node_attributes(G, 'pos')
                    folium.Marker(
                        location=[pos[delivery_person_location][0], pos[delivery_person_location][1]],
                        popup=f"Delivery Person: {delivery_person_location}",
                        icon=folium.Icon(color='green', icon='user', prefix='fa')
                    ).add_to(m)
                    
                    # Add numbered markers for each stop in the route
                    for i, loc in enumerate(route):
                        # Create a circle marker with the stop number
                        folium.CircleMarker(
                            location=[pos[loc][0], pos[loc][1]],
                            radius=15,
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=0.8,
                            popup=f"Stop {i+1}: {loc}"
                        ).add_to(m)
                        
                        # Add the stop number as text
                        folium.Marker(
                            location=[pos[loc][0], pos[loc][1]],
                            icon=DivIcon(
                                icon_size=(20, 20),
                                icon_anchor=(10, 10),
                                html=f'<div style="font-size: 10pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>'
                            )
                        ).add_to(m)
                    
                    # Add path from delivery person to first stop
                    if first_stop_path and len(first_stop_path) > 1:
                        for i in range(len(first_stop_path)-1):
                            u, v = first_stop_path[i], first_stop_path[i+1]
                            lat1, lon1 = pos[u]
                            lat2, lon2 = pos[v]
                            folium.PolyLine(
                                locations=[[lat1, lon1], [lat2, lon2]],
                                color='green',
                                weight=4,
                                opacity=0.8,
                                popup=f"To first stop: {u} to {v}"
                            ).add_to(m)
            
            else:  # Road-based routing
                # Get the road network for Marrakech
                road_G = get_road_network("Marrakech, Morocco", "drive")
                
                if road_G is None:
                    st.error("Failed to get road network. Please try again later.")
                else:
                    # Convert location names to coordinates
                    location_coords = [locations_dict[loc] for loc in selected_locations]
                    delivery_person_coords = locations_dict[delivery_person_location]
                    
                    # Create distance matrix using road-based approach
                    weight = 'length' if cost_type_tsp == 'distance' else 'travel_time'
                    distance_matrix, paths_dict = create_distance_matrix_on_road(road_G, location_coords, weight)
                    
                    # Solve TSP with the road-based distance matrix
                    route_indices = solve_tsp(distance_matrix, delivery_person_location=delivery_person_location, locations=selected_locations)
                    
                    if route_indices:
                        route = [selected_locations[i] for i in route_indices]
                        
                        # Calculate total cost and collect all path coordinates
                        total_cost = 0
                        all_path_coords = []
                        all_path_nodes = []
                        
                        for i in range(len(route)-1):
                            start_idx = route_indices[i]
                            end_idx = route_indices[i+1]
                            
                            if (start_idx, end_idx) in paths_dict:
                                path_nodes, path_coords = paths_dict[(start_idx, end_idx)]
                                total_cost += distance_matrix[start_idx][end_idx]
                                
                                # Add path nodes and coordinates
                                if i < len(route)-2:
                                    # Add all nodes except the last to avoid duplicates
                                    all_path_nodes.extend(path_nodes[:-1])
                                    
                                    # For coordinates, we need to be careful about connections
                                    if all_path_coords and path_coords:
                                        # Check if the last point in all_path_coords is the same as the first in path_coords
                                        last_point = all_path_coords[-1]
                                        first_point = path_coords[0]
                                        
                                        # If they're the same (or very close), skip the first point in path_coords
                                        if (abs(last_point[0] - first_point[0]) < 1e-6 and 
                                            abs(last_point[1] - first_point[1]) < 1e-6):
                                            all_path_coords.extend(path_coords[1:])
                                        else:
                                            # If there's a gap between segments, try to find the nearest node
                                            # and add intermediate points to create a smooth connection
                                            distance = ((last_point[0] - first_point[0])**2 + 
                                                        (last_point[1] - first_point[1])**2)**0.5
                                            
                                            # If the gap is small enough, we can just connect directly
                                            if distance < 0.001:  # Approximately 100m
                                                all_path_coords.extend(path_coords)
                                            else:
                                                # For larger gaps, we should try to find a connecting path
                                                # but for now, just add both points to create a visible connection
                                                all_path_coords.extend(path_coords)
                                    else:
                                        all_path_coords.extend(path_coords)
                                else:
                                    # For the last segment, add all nodes and coordinates
                                    all_path_nodes.extend(path_nodes)
                                    all_path_coords.extend(path_coords)
                        
                        # Calculate path from delivery person to first stop
                        delivery_node = get_nearest_node(road_G, delivery_person_coords)
                        first_stop_coords = locations_dict[route[0]]
                        first_stop_path, first_stop_cost, first_stop_coords = find_shortest_path_on_road(
                            road_G, delivery_person_coords, first_stop_coords, weight
                        )
                        
                        # Display results
                        unit = 'km' if cost_type_tsp == 'distance' else 'minutes'
                        # Format distance with 3 decimal places if it's in kilometers
                        if cost_type_tsp == 'distance':
                            formatted_cost = f"{total_cost:.3f}"
                        else:
                            formatted_cost = f"{total_cost}"
                        st.success(f"Optimal route found! Total {cost_type_tsp}: {formatted_cost} {unit}")
                        # Format first stop distance with 3 decimal places if it's in kilometers
                        if cost_type_tsp == 'distance':
                            formatted_first_stop_cost = f"{first_stop_cost:.3f}"
                        else:
                            formatted_first_stop_cost = f"{first_stop_cost}"
                        st.write(f"Distance from {delivery_person_location} to first stop: {formatted_first_stop_cost} {unit}")
                        st.write("Route:")
                        for i, loc in enumerate(route):
                            st.write(f"{i+1}. {loc}")
                        
                        # Create markers for the map
                        markers = []
                        # Add delivery person marker
                        markers.append((delivery_person_coords[0], delivery_person_coords[1], f"Delivery Person: {delivery_person_location}"))
                        
                        # Add markers for each stop
                        for i, loc in enumerate(route):
                            lat, lon = locations_dict[loc]
                            markers.append((lat, lon, f"Stop {i+1}: {loc}"))
                        
                        # Create the map using road_map functions
                        m = create_road_folium_map(road_G, all_path_coords, markers, all_path_nodes)
                        
                        # Add path from delivery person to first stop with a different color
                        if first_stop_coords and len(first_stop_coords) > 1:
                            folium.PolyLine(
                                locations=first_stop_coords,
                                color='green',
                                weight=4,
                                opacity=0.8,
                                popup="Path to first stop"
                            ).add_to(m)
                        
                        # Add numbered markers for each stop
                        for i, loc in enumerate(route):
                            lat, lon = locations_dict[loc]
                            # Create a circle marker with the stop number
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=18,  # Slightly larger radius
                                color='red',
                                fill=True,
                                fill_color='red',
                                fill_opacity=0.9,  # More opaque
                                popup=f"Stop {i+1}: {loc}"
                            ).add_to(m)
                            
                            # Add the stop number as text
                            folium.Marker(
                                location=[lat, lon],
                                icon=DivIcon(
                                    icon_size=(20, 20),
                                    icon_anchor=(10, 10),
                                    html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>'
                                )
                            ).add_to(m)
                            
                            # Add a tooltip that appears on hover
                            folium.Tooltip(
                                f"Stop {i+1}: {loc}"
                            ).add_to(m)
                
                st.subheader("Interactive Map")
                st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
    elif len(selected_locations) <= 1:
        with col2:
            st.warning("Please select at least 2 locations.")

# Add footer
st.markdown("---")
st.markdown("### How to use this app")
st.markdown("""
1. **Shortest Path**: Select a starting point and destination to find the shortest route between them.
2. **Multi-Stop Route**: Select multiple locations to visit, and the app will find the optimal order to visit them all.
   - **Graph-based routing**: Uses a simplified graph model with direct connections between locations.
   - **Road-based routing**: Uses actual road networks for more realistic routes that follow real streets.

The interactive map shows the route with red lines and markers for each location.
""")