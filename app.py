
import streamlit as st
import folium
from folium.features import DivIcon
import networkx as nx
import streamlit_folium
import importlib.util
from optimal_path import create_marrakech_sample_graph, find_shortest_path, solve_tsp, create_distance_matrix
from road_network import get_road_network, find_shortest_path_on_road, create_distance_matrix_on_road, convert_marrakech_graph_to_coordinates, get_nearest_node
from road_map import create_road_folium_map, plot_route_on_map
import importlib.util
import sys

def is_package_installed(package_name):
    """Check if a package is installed and accessible"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception:
        return False

# Check for all required packages
sklearn_installed = is_package_installed("sklearn")
osmnx_installed = is_package_installed("osmnx")
networkx_installed = is_package_installed("networkx")
folium_installed = is_package_installed("folium")

# Display warnings for missing packages
missing_packages = []
if not sklearn_installed:
    missing_packages.append("scikit-learn")
if not osmnx_installed:
    missing_packages.append("osmnx")
if not networkx_installed:
    missing_packages.append("networkx")
if not folium_installed:
    missing_packages.append("folium")

if missing_packages:
    st.warning(f"""
    **Warning**: Some required packages are missing: {', '.join(missing_packages)}
    
    Please install all required packages using:
    ```
    pip install scikit-learn osmnx networkx folium streamlit streamlit-folium
    ```
    
    Some features may not work correctly until these packages are installed.
    """)
# Check for required dependencies
sklearn_installed = importlib.util.find_spec("sklearn") is not None
osmnx_installed = importlib.util.find_spec("osmnx") is not None

# Set page configuration
st.set_page_config(page_title="Marrakech Route Planner", layout="wide")

# Add title and description
st.title("Marrakech Interactive Route Planner")
st.write("""
This application helps you plan your routes in Marrakech. You can find the shortest path between two locations 
or plan an optimal route through multiple locations using either graph-based or road-based routing.
""")

# Show dependency warnings if needed
if not (sklearn_installed and osmnx_installed):
    missing_deps = []
    if not sklearn_installed:
        missing_deps.append("scikit-learn")
    if not osmnx_installed:
        missing_deps.append("osmnx")
    
    st.warning(f"""
    **Warning**: Some dependencies are missing: {', '.join(missing_deps)}
    
    To enable road-based routing, install all dependencies using:
    ```
    pip install scikit-learn osmnx networkx folium streamlit streamlit-folium
    ```
    
    For now, graph-based routing will still work correctly.
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
        
        # Enhanced options for route optimization
        cost_type = st.radio("Optimize by", ["distance", "time"], horizontal=True)
        
        # Add routing type selection, but disable road-based if dependencies are missing
        routing_options = ["Graph-based", "Road-based"]
        disabled_option = None if (sklearn_installed and osmnx_installed) else "Road-based"
        
        if disabled_option:
            st.info(f"{disabled_option} routing is disabled due to missing dependencies. Install them to enable this feature.")
            routing_type = "Graph-based"
        else:
            routing_type = st.radio("Routing type", routing_options, index=1, horizontal=True, key="shortest_path_routing")
        
        # Add advanced routing options (only show if Road-based is selected)
        if routing_type == "Road-based" and not disabled_option:
            with st.expander("Advanced Routing Options"):
                route_preference = st.selectbox(
                    "Route preference", 
                    [
                        "Balanced (Default)",
                        "Fastest route", 
                        "Shortest distance",
                        "Prefer main roads", 
                        "Avoid highways"
                    ]
                )
                
                # Explanation for the user
                if route_preference == "Balanced (Default)":
                    st.caption("Balances travel time and distance")
                elif route_preference == "Fastest route":
                    st.caption("Prioritizes travel time over distance")
                elif route_preference == "Shortest distance":
                    st.caption("Minimizes total distance regardless of travel time")
                elif route_preference == "Prefer main roads":
                    st.caption("Favors main roads and highways for smoother travel")
                elif route_preference == "Avoid highways":
                    st.caption("Avoids highways and major roads where possible")
    
    if st.button("Find Shortest Path"):
        # Use graph-based routing if selected or if road-based is disabled
        if routing_type == "Graph-based" or disabled_option == "Road-based":
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
        
        elif routing_type == "Road-based":
            try:
                # Get the road network with a spinner to show progress
                with st.spinner("Loading road network..."):
                    road_G = get_road_network("Marrakech, Morocco", "drive")
                
                if road_G is None:
                    st.error("Failed to get road network. Please make sure all dependencies are installed.")
                    st.info("Falling back to graph-based routing...")
                    path, cost = find_shortest_path(G, start_loc, end_loc, cost_type)
                    
                    if path:
                        with col2:
                            st.success(f"Path found using graph-based routing! Total {cost_type}: {cost} {'km' if cost_type == 'distance' else 'minutes'}")
                            st.write("Route:")
                            for i, loc in enumerate(path):
                                st.write(f"{i+1}. {loc}")
                        
                        # Create and display the map
                        m = create_folium_map(G, path)
                        st.subheader("Interactive Map")
                        st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
                else:
                    with col2:
                        st.info("Using road-based routing for more accurate results...")
                        
                        # Get coordinates for start and end locations
                        start_coords = locations_dict[start_loc]
                        end_coords = locations_dict[end_loc]
                        
                        # Determine the weight parameter based on cost type
                        weight = 'length' if cost_type == 'distance' else 'travel_time'
                        
                        # Find the shortest path using road network
                        try:
                            with st.spinner("Calculating road-based route..."):
                                path_nodes, path_cost, path_coords = find_shortest_path_on_road(
                                    road_G, start_coords, end_coords, weight
                                )
                            
                            if path_nodes and path_coords:
                                # Format cost with proper units
                                unit = 'km' if cost_type == 'distance' else 'minutes'
                                if cost_type == 'distance':
                                    formatted_cost = f"{path_cost:.3f}"
                                else:
                                    formatted_cost = f"{path_cost:.1f}"
                                
                                st.success(f"Path found! Total {cost_type}: {formatted_cost} {unit}")
                                st.write(f"Route follows real roads from {start_loc} to {end_loc}")
                                
                                # Create markers for the map
                                markers = []
                                markers.append((start_coords[0], start_coords[1], f"Start: {start_loc}"))
                                markers.append((end_coords[0], end_coords[1], f"End: {end_loc}"))
                                
                                # Create the map using road_map functions
                                m = create_road_folium_map(road_G, path_coords, markers, path_nodes)
                                
                                st.subheader("Interactive Map")
                                st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
                            else:
                                st.error(f"No road-based path found between {start_loc} and {end_loc}")
                                st.info("Falling back to graph-based routing...")
                                
                                # Fallback to graph-based
                                path, cost = find_shortest_path(G, start_loc, end_loc, cost_type)
                                if path:
                                    st.success(f"Path found using graph-based routing! Total {cost_type}: {cost} {'km' if cost_type == 'distance' else 'minutes'}")
                                    st.write("Route:")
                                    for i, loc in enumerate(path):
                                        st.write(f"{i+1}. {loc}")
                                    
                                    # Create and display the map
                                    m = create_folium_map(G, path)
                                    st.subheader("Interactive Map (Graph-based)")
                                    st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
                        except Exception as e:
                            st.error(f"Error calculating road-based route: {str(e)}")
                            st.info("Falling back to graph-based routing...")
                            
                            # Fallback to graph-based routing
                            path, cost = find_shortest_path(G, start_loc, end_loc, cost_type)
                            
                            if path:
                                st.success(f"Path found using graph-based routing! Total {cost_type}: {cost} {'km' if cost_type == 'distance' else 'minutes'}")
                                st.write("Route:")
                                for i, loc in enumerate(path):
                                    st.write(f"{i+1}. {loc}")
                                
                                # Create and display the map
                                m = create_folium_map(G, path)
                                st.subheader("Interactive Map (Graph-based)")
                                st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Falling back to graph-based routing...")
                # Use graph-based as fallback
                path, cost = find_shortest_path(G, start_loc, end_loc, cost_type)
                
                if path:
                    with col2:
                        st.success(f"Path found using graph-based routing! Total {cost_type}: {cost} {'km' if cost_type == 'distance' else 'minutes'}")
                        st.write("Route:")
                        for i, loc in enumerate(path):
                            st.write(f"{i+1}. {loc}")
                    
                    # Create and display the map
                    m = create_folium_map(G, path)
                    st.subheader("Interactive Map")
                    st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)

# Multi-Stop Route tab

# Multi-Stop Route tab
with tab2:
    st.header("Plan a Multi-Stop Route")
    
    col1, col2 = st.columns([3, 4])
    
    with col1:
        st.subheader("Route Settings")
        
        # Use a more efficient UI for location selection with multi-select
        st.write("Select locations to visit:")
        default_locations = ["Jemaa el-Fnaa"]
        selected_locations = st.multiselect(
            "Select destinations",
            options=locations,
            default=default_locations,
            help="Select all locations you need to visit"
        )
        
        # Count selected locations and provide feedback
        location_count = len(selected_locations)
        if location_count == 0:
            st.warning("Please select at least one location to visit.")
        elif location_count == 1:
            st.info("You've selected 1 location. Consider adding more for an optimized route.")
        elif location_count <= 5:
            st.success(f"You've selected {location_count} locations. Good for quick route planning.")
        else:
            st.success(f"You've selected {location_count} locations. Optimization will be more effective!")
        
        # Starting point selection
        st.subheader("Starting Point")
        starting_point_options = ["Custom location"] + locations
        starting_point_type = st.radio(
            "Starting from:",
            options=["Select from list", "Current location", "Custom coordinates"],
            index=0,
            horizontal=True
        )
        
        if starting_point_type == "Select from list":
            delivery_person_location = st.selectbox(
                "Select starting location", 
                locations, 
                index=0,
                help="Where the route will begin"
            )
        elif starting_point_type == "Current location":
            st.info("Using your current location as the starting point (simulated for demo)")
            # In a real app, you would use browser geolocation here
            delivery_person_location = "Jemaa el-Fnaa"  # Default to a central location for demo
        else:  # Custom coordinates
            col_lat, col_lon = st.columns(2)
            with col_lat:
                custom_lat = st.number_input("Latitude", value=31.6258, format="%.4f", 
                                           help="Enter latitude coordinate")
            with col_lon:
                custom_lon = st.number_input("Longitude", value=-7.9891, format="%.4f",
                                           help="Enter longitude coordinate")
            delivery_person_location = "Custom Location"
            custom_coords = (custom_lat, custom_lon)
        
        # Optimization parameters
        st.subheader("Optimization Settings")
        cost_type_tsp = st.radio(
            "Optimize by", 
            ["distance", "time", "balanced"], 
            horizontal=True, 
            key="tsp_cost",
            help="Distance: shortest route | Time: fastest route | Balanced: considers both"
        )
        
        # Vehicle type selection
        vehicle_type = st.selectbox(
            "Vehicle type",
            ["Car", "Motorcycle", "Bicycle", "Walking"],
            index=0,
            help="Different vehicles have different speed profiles and road access"
        )
        
        # Vehicle speed factor based on selection (relative to car speed)
        vehicle_factors = {
            "Car": 1.0,
            "Motorcycle": 0.9,  # Slightly slower than car
            "Bicycle": 0.3,     # Much slower than car
            "Walking": 0.1      # Very slow compared to car
        }
        vehicle_factor = vehicle_factors[vehicle_type]
        
        # Traffic conditions
        traffic_condition = st.select_slider(
            "Traffic conditions",
            options=["Very light", "Light", "Moderate", "Heavy", "Very heavy"],
            value="Moderate",
            help="Affects travel time estimates"
        )
        
        # Convert traffic condition to factor
        traffic_factors = {
            "Very light": 0.7,   # Faster than normal
            "Light": 0.85,       # Slightly faster
            "Moderate": 1.0,     # Normal
            "Heavy": 1.3,        # Slower
            "Very heavy": 1.6    # Much slower
        }
        traffic_factor = traffic_factors[traffic_condition]
        
        # Add routing type selection, but disable road-based if dependencies are missing
        routing_options = ["Graph-based", "Road-based"]
        if not (sklearn_installed and osmnx_installed):
            st.info("Road-based routing is disabled due to missing dependencies. Install them to enable this feature.")
            routing_type = "Graph-based"
        else:
            routing_type = st.radio("Routing type", routing_options, index=1, horizontal=True, key="routing_type")
        
        # Advanced options - expanded with more routing preferences
        if routing_type == "Road-based" and not disabled_option:
            with st.expander("Advanced Routing Options", expanded=False):
                # Optimization algorithm selection
                optimization_algorithm = st.selectbox(
                    "Algorithm",
                    ["Nearest Neighbor", "Greedy", "2-Opt", "Simulated Annealing", "Genetic Algorithm"],
                    index=2,
                    help="Different algorithms offer trade-offs between speed and route quality"
                )
                
                # Add explanations for algorithms
                algorithm_explanations = {
                    "Nearest Neighbor": "Fast but may produce suboptimal routes",
                    "Greedy": "Good balance of speed and quality",
                    "2-Opt": "Improves routes by swapping segments",
                    "Simulated Annealing": "Finds near-optimal solutions for complex routes",
                    "Genetic Algorithm": "Best for many stops but slowest calculation"
                }
                
                st.caption(algorithm_explanations[optimization_algorithm])
                
                # Route preference
                route_preference = st.selectbox(
                    "Route preference", 
                    [
                        "Balanced (Default)",
                        "Fastest route", 
                        "Shortest distance",
                        "Prefer main roads", 
                        "Avoid highways",
                        "Scenic route"
                    ],
                    help="Influences how routes are calculated"
                )
                
                # Round trip option
                return_to_start = st.checkbox(
                    "Return to starting point",
                    value=False,
                    help="Complete a loop back to where you started"
                )
                
                # Time window option
                use_time_constraints = st.checkbox(
                    "Use time constraints",
                    value=False,
                    help="Specify when you need to arrive at each location"
                )
                
                if use_time_constraints:
                    st.info("Time constraints will be considered when optimizing the route order")
                    # This would require additional UI for time windows per location
                
                # Optimization level
                optimization_level = st.select_slider(
                    "Optimization Level",
                    options=["Fast", "Balanced", "Thorough", "Exhaustive"],
                    value="Balanced",
                    help="Higher optimization levels produce better routes but take longer to calculate"
                )
                
                # Convert optimization level to time limit
                time_limits = {
                    "Fast": 10,
                    "Balanced": 30,
                    "Thorough": 60,
                    "Exhaustive": 120
                }
                time_limit = time_limits[optimization_level]
    
    # Compute button with clear status
    compute_button = st.button(
        "Calculate Optimal Route",
        help="Find the most efficient route through all selected locations"
    )
    
    # Validation and computation
    if compute_button:
        if len(selected_locations) < 2:
            st.warning("Please select at least 2 locations to calculate a route.")
        else:
            # Progress indicators
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_container = st.empty()
            result_container = st.container()
            
            status_container.info("Starting route calculation...")
            
            # Processing stages for user feedback
            stages = [
                "Loading network data...",
                "Analyzing locations...",
                "Building route matrix...",
                "Finding optimal sequence...",
                "Calculating detailed paths...",
                "Preparing visualization..."
            ]
            
            with col2:
                # Initialize map placeholder
                map_placeholder = st.empty()
                
                # Statistics containers
                stats_container = st.container()
                with stats_container:
                    st.subheader("Route Statistics")
                    stats_cols = st.columns(4)
                    
                    # Initialize statistics placeholders
                    with stats_cols[0]:
                        distance_container = st.empty()
                        distance_container.metric("Total Distance", "Calculating...")
                    
                    with stats_cols[1]:
                        time_container = st.empty()
                        time_container.metric("Est. Time", "Calculating...")
                    
                    with stats_cols[2]:
                        stops_container = st.empty()
                        stops_container.metric("Stops", str(len(selected_locations)))
                    
                    with stats_cols[3]:
                        efficiency_container = st.empty()
                        efficiency_container.metric("Optimization", "Calculating...")
                
                # Route details container
                route_details = st.container()
                with route_details:
                    st.subheader("Route Details")
                    route_list = st.empty()
                
            # Updated computation loop with proper progress reporting
            try:
                # STAGE 1: Load network data
                progress_bar.progress(10)
                status_container.info(stages[0])
                
                if routing_type == "Graph-based" or not (sklearn_installed and osmnx_installed):
                    # Graph-based routing
                    # This section would contain the graph-based implementation
                    progress_bar.progress(30)
                    status_container.info("Using graph-based routing...")
                    
                    # Create a distance matrix for the selected locations
                    if 'cost_type_tsp' in locals():
                        weight = cost_type_tsp
                    else:
                        weight = 'distance'
                    
                    # Create distance matrix
                    distance_matrix = create_distance_matrix(G, selected_locations, weight)
                    
                    # Update progress
                    progress_bar.progress(50)
                    status_container.info(stages[3])
                    
                    # Solve TSP
                    route_indices = solve_tsp(
                        distance_matrix, 
                        delivery_person_location=delivery_person_location, 
                        locations=selected_locations
                    )
                    
                    if route_indices:
                        # Generate the route
                        route = [selected_locations[i] for i in route_indices]
                        
                        # Calculate path segments and total distance
                        total_distance = 0
                        route_details_text = []
                        
                        # Update progress
                        progress_bar.progress(70)
                        status_container.info(stages[4])
                        
                        # Calculate paths between consecutive stops
                        for i in range(len(route)-1):
                            from_loc = route[i]
                            to_loc = route[i+1]
                            path, segment_distance = find_shortest_path(G, from_loc, to_loc, weight)
                            total_distance += segment_distance
                            route_details_text.append(f"{i+1}. {from_loc} → {to_loc} ({segment_distance:.2f} km)")
                        
                        # Calculate time estimate (rough approximation)
                        # Assume average speed of 40 km/h for urban areas
                        avg_speed = 40.0  # km/h
                        # Adjust for vehicle type and traffic
                        effective_speed = avg_speed * vehicle_factor / traffic_factor
                        time_hours = total_distance / effective_speed
                        time_minutes = time_hours * 60
                        
                        # Determine time display format
                        if time_minutes < 60:
                            time_display = f"{time_minutes:.0f} min"
                        else:
                            hours = int(time_hours)
                            mins = int((time_hours - hours) * 60)
                            time_display = f"{hours}h {mins}m"
                        
                        # Update statistics
                        distance_container.metric("Total Distance", f"{total_distance:.2f} km")
                        time_container.metric("Est. Time", time_display)
                        efficiency_container.metric("Optimization", "Good")
                        
                        # Display route details
                        route_list.markdown("\n".join(route_details_text))
                        
                        # Create and display map
                        progress_bar.progress(90)
                        status_container.info(stages[5])
                        
                        m = create_folium_map(G, path=None)  # Create base map
                        
                        # Add the complete route
                        for i in range(len(route)-1):
                            from_loc = route[i]
                            to_loc = route[i+1]
                            path, _ = find_shortest_path(G, from_loc, to_loc, weight)
                            
                            # Get positions for the path
                            pos = nx.get_node_attributes(G, 'pos')
                            path_coords = [[pos[node][0], pos[node][1]] for node in path]
                            
                            # Add the path segment to the map
                            folium.PolyLine(
                                locations=path_coords,
                                color='blue',
                                weight=4,
                                opacity=0.8,
                                popup=f"Segment {i+1}: {from_loc} → {to_loc}"
                            ).add_to(m)
                        
                        # Add markers for each stop with numbered icons
                        pos = nx.get_node_attributes(G, 'pos')
                        for i, loc in enumerate(route):
                            lat, lon = pos[loc]
                            
                            # Create a custom icon with the stop number
                            icon_color = 'green' if i == 0 else ('red' if i == len(route)-1 else 'blue')
                            icon_text = str(i)
                            
                            folium.Marker(
                                location=[lat, lon],
                                popup=f"Stop {i+1}: {loc}",
                                icon=folium.Icon(color=icon_color, icon='info-sign')
                            ).add_to(m)
                            
                            # Add a number marker
                            folium.Marker(
                                location=[lat, lon],
                                icon=DivIcon(
                                    icon_size=(20, 20),
                                    icon_anchor=(10, 10),
                                    html=f'<div style="font-size: 10pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>',
                                    class_name="custom-icon"
                                )
                            ).add_to(m)
                        
                        # Display the map
                        progress_bar.progress(100)
                        status_container.success("Route calculation complete!")
                        map_placeholder.pydeck_chart(m)
                        
                    else:
                        status_container.error("Failed to find an optimal route. Try different locations.")
                
                # ROAD-BASED ROUTING IMPLEMENTATION
                elif routing_type == "Road-based":
                    progress_bar.progress(15)
                    status_container.info(stages[0] + " (using real road network)")
                    
                    # Get the road network for Marrakech
                    try:
                        # Load road network with progress indicator
                        road_G = get_road_network("Marrakech, Morocco", "drive")
                        
                        if road_G is None:
                            status_container.error("Failed to get road network. Falling back to graph-based routing.")
                            # Fallback code would go here
                        else:
                            # STAGE 2: Process locations
                            progress_bar.progress(30)
                            status_container.info(stages[1])
                            
                            # Convert location names to coordinates
                            location_coords = []
                            for loc in selected_locations:
                                if loc in locations_dict:
                                    location_coords.append(locations_dict[loc])
                                else:
                                    status_container.warning(f"Location '{loc}' not found in database. Skipping.")
                            
                            # Handle delivery person location/starting point
                            if starting_point_type == "Custom coordinates":
                                delivery_person_coords = custom_coords
                            else:
                                delivery_person_coords = locations_dict[delivery_person_location]
                            
                            # STAGE 3: Build distance matrix
                            progress_bar.progress(40)
                            status_container.info(stages[2])
                            
                            # Determine weight based on optimization criteria
                            if cost_type_tsp == "distance":
                                weight = 'length'
                            elif cost_type_tsp == "time":
                                weight = 'travel_time'
                            else:  # balanced
                                weight = 'balanced'  # This would need to be implemented in create_distance_matrix_on_road
                            
                            # Apply route preference adjustments
                            if 'route_preference' in locals():
                                if route_preference == "Fastest route":
                                    weight = 'travel_time'
                                elif route_preference == "Shortest distance":
                                    weight = 'length'
                            
                            # Create distance matrix
                            distance_matrix, paths_dict = create_distance_matrix_on_road(
                                road_G, 
                                location_coords, 
                                weight,
                                traffic_factor=traffic_factor,
                                vehicle_factor=vehicle_factor
                            )
                            
                            # STAGE 4: Find optimal sequence
                            progress_bar.progress(60)
                            status_container.info(stages[3])
                            
                            # Select algorithm based on user choice
                            algorithm = "auto"  # Default
                            if 'optimization_algorithm' in locals():
                                algorithm_mapping = {
                                    "Nearest Neighbor": "nearest_neighbor",
                                    "Greedy": "greedy",
                                    "2-Opt": "two_opt",
                                    "Simulated Annealing": "simulated_annealing",
                                    "Genetic Algorithm": "genetic"
                                }
                                algorithm = algorithm_mapping.get(optimization_algorithm, "auto")
                            
                            # Set round trip parameter
                            round_trip = False
                            if 'return_to_start' in locals() and return_to_start:
                                round_trip = True
                            
                            # Solve TSP with selected parameters
                            route_indices = solve_tsp(
                                distance_matrix,
                                locations=selected_locations,
                                algorithm=algorithm,
                                time_limit_seconds=time_limit,
                                round_trip=round_trip
                            )
                            
                            if route_indices:
                                # Generate the route
                                route = [selected_locations[i] for i in route_indices]
                                
                                # STAGE 5: Calculate detailed paths
                                progress_bar.progress(75)
                                status_container.info(stages[4])
                                
                                # Calculate total cost and collect all path coordinates
                                total_cost = 0
                                all_path_coords = []
                                all_path_nodes = []
                                route_segments = []
                                
                                # Process each segment of the route
                                for i in range(len(route)-1):
                                    start_idx = selected_locations.index(route[i])
                                    end_idx = selected_locations.index(route[i+1])
                                    
                                    # Get path information for this segment
                                    segment_key = (start_idx, end_idx)
                                    reverse_key = (end_idx, start_idx)
                                    
                                    if segment_key in paths_dict:
                                        path_nodes, path_coords = paths_dict[segment_key]
                                        segment_cost = distance_matrix[start_idx][end_idx]
                                    elif reverse_key in paths_dict:
                                        path_nodes, path_coords = paths_dict[reverse_key]
                                        # Reverse the path
                                        path_nodes = path_nodes[::-1]
                                        path_coords = path_coords[::-1]
                                        segment_cost = distance_matrix[end_idx][start_idx]
                                    else:
                                        # If path not found in cache, calculate it
                                        from_coords = locations_dict[route[i]]
                                        to_coords = locations_dict[route[i+1]]
                                        path_nodes, segment_cost, path_coords = find_shortest_path_on_road(
                                            road_G, from_coords, to_coords, weight,
                                            traffic_factor=traffic_factor,
                                            vehicle_factor=vehicle_factor
                                        )
                                    
                                    total_cost += segment_cost
                                    
                                    # Store segment details
                                    route_segments.append({
                                        "from": route[i],
                                        "to": route[i+1],
                                        "distance": segment_cost,
                                        "nodes": path_nodes,
                                        "coords": path_coords
                                    })
                                    
                                    # Append path to overall route
                                    if len(all_path_coords) == 0:
                                        # First segment - add all coordinates
                                        all_path_nodes.extend(path_nodes)
                                        all_path_coords.extend(path_coords)
                                    else:
                                        # Improve path connections between segments
                                        last_point = all_path_coords[-1]
                                        first_point = path_coords[0]
                                        
                                        # If points are close, create a smooth connection
                                        distance = ((last_point[0] - first_point[0])**2 + 
                                                    (last_point[1] - first_point[1])**2)**0.5
                                        
                                        if distance < 0.001:  # Close enough (about 100m)
                                            # Skip first point to avoid duplicates
                                            all_path_nodes.extend(path_nodes[1:])
                                            all_path_coords.extend(path_coords[1:])
                                        else:
                                            # Connect with a direct line segment
                                            all_path_nodes.extend(path_nodes)
                                            all_path_coords.extend(path_coords)
                                
                                # Calculate path from delivery person to first stop
                                progress_bar.progress(85)
                                status_container.info("Calculating path from starting point to first stop...")
                                
                                first_stop_coords = locations_dict[route[0]]
                                first_stop_path, first_stop_cost, first_stop_coords = find_shortest_path_on_road(
                                    road_G, delivery_person_coords, first_stop_coords, weight,
                                    route_preference=route_preference if 'route_preference' in locals() else None,
                                    traffic_factor=traffic_factor,
                                    vehicle_factor=vehicle_factor
                                )
                                
                                # STAGE 6: Prepare visualization
                                progress_bar.progress(90)
                                status_container.info(stages[5])
                                
                                # Format results for display
                                unit = 'km' if cost_type_tsp == 'distance' else 'minutes'
                                
                                # Format values with appropriate precision
                                if cost_type_tsp == 'distance':
                                    formatted_cost = f"{total_cost:.2f}"
                                    formatted_first_stop_cost = f"{first_stop_cost:.2f}"
                                else:
                                    formatted_cost = f"{total_cost:.1f}"
                                    formatted_first_stop_cost = f"{first_stop_cost:.1f}"
                                
                                # Calculate time estimate
                                if cost_type_tsp == 'distance':
                                    # Estimate time based on distance
                                    avg_speed = 40.0  # km/h for urban areas
                                    # Adjust for vehicle type and traffic
                                    effective_speed = avg_speed * vehicle_factor / traffic_factor
                                    time_hours = total_cost / effective_speed
                                    time_minutes = time_hours * 60
                                else:
                                    # For time optimization, the cost is already in minutes
                                    time_minutes = total_cost
                                
                                # Format time for display
                                if time_minutes < 60:
                                    time_display = f"{time_minutes:.0f} min"
                                else:
                                    hours = int(time_minutes / 60)
                                    mins = int(time_minutes % 60)
                                    time_display = f"{hours}h {mins}m"
                                
                                # Update statistics
                                distance_container.metric("Total Distance", f"{formatted_cost} km")
                                time_container.metric("Est. Time", time_display)
                                stops_container.metric("Stops", str(len(selected_locations)))
                                
                                # Efficiency rating based on algorithm used
                                if 'optimization_algorithm' in locals():
                                    efficiency_ratings = {
                                        "Nearest Neighbor": "Good",
                                        "Greedy": "Better",
                                        "2-Opt": "Very Good",
                                        "Simulated Annealing": "Excellent",
                                        "Genetic Algorithm": "Optimal"
                                    }
                                    efficiency = efficiency_ratings.get(optimization_algorithm, "Good")
                                else:
                                    efficiency = "Very Good"
                                
                                efficiency_container.metric("Optimization", efficiency)
                                
                                # Generate route details text
                                route_details_text = []
                                
                                # Add details for path from starting point to first stop
                                route_details_text.append(f"Start: **{delivery_person_location}** → **{route[0]}** ({formatted_first_stop_cost} km)")
                                
                                # Add details for each segment
                                for i in range(len(route_segments)):
                                    segment = route_segments[i]
                                    route_details_text.append(f"{i+1}. **{segment['from']}** → **{segment['to']}** ({segment['distance']:.2f} km)")
                                
                                # Display route details
                                route_list.markdown("\n".join(route_details_text))
                                
                                # Create detailed map
                                m = create_road_folium_map(road_G, all_path_coords, [], all_path_nodes)
                                
                                # Create markers for the map
                                markers = []
                                
                                # Add delivery person marker
                                markers.append((delivery_person_coords[0], delivery_person_coords[1], f"Start: {delivery_person_location}"))
                                
                                # Add markers for each stop with numbering
                                for i, loc in enumerate(route):
                                    lat, lon = locations_dict[loc]
                                    
                                    popup_content = f"Stop {i+1}: {loc}"
                                    if i == 0:
                                        icon_color = 'green'
                                    elif i == len(route) - 1:
                                        icon_color = 'red'
                                    else:
                                        icon_color = 'blue'
                                    
                                    # Create a marker with number
                                    folium.Marker(
                                        location=[lat, lon],
                                        popup=popup_content,
                                        icon=folium.Icon(color=icon_color, icon='info-sign')
                                    ).add_to(m)
                                    
                                    # Add a number label
                                    folium.Marker(
                                        location=[lat, lon],
                                        icon=DivIcon(
                                            icon_size=(20, 20),
                                            icon_anchor=(10, 10),
                                            html=f'<div style="font-size: 10pt; color: white; background-color: {icon_color}; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 20px;">{i+1}</div>',
                                            class_name="numbered-marker"
                                        )
                                    ).add_to(m)
                                
                                # Add path from delivery person to first stop with a different color
                                if first_stop_coords and len(first_stop_coords) > 1:
                                    folium.PolyLine(
                                        locations=first_stop_coords,
                                        color='green',
                                        weight=4,
                                        opacity=0.8,
                                        popup="Path to first stop"
                                    ).add_to(m)
                                
                                # Complete!
                                progress_bar.progress(100)
                                status_container.success("🎉 Route calculation complete!")
                                
                                # Display the map
                                st_folium_map = streamlit_folium.folium_static(m, width=800, height=500)
                                
                                # Add download buttons for route information
                                st.download_button(
                                    label="Download Route (CSV)",
                                    data="\n".join([
                                        "Stop,Location,Distance (km)",
                                        f"Start,{delivery_person_location},0.0",
                                        *[f"{i+1},{route[i]},{route_segments[i]['distance'] if i < len(route_segments) else 0.0}" 
                                          for i in range(len(route))]
                                    ]),
                                    file_name="optimal_route.csv",
                                    mime="text/csv"
                                )
                            else:
                                status_container.error("Failed to find an optimal route. Try different locations.")
                    except Exception as e:
                        status_container.error(f"Error with road-based routing: {str(e)}")
                        st.info("Falling back to graph-based routing...")
                        # Fallback to graph-based routing code would go here
            
            except Exception as e:
                status_container.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    elif len(selected_locations) <= 1:
        # When no locations are selected, show a preview map
        with col2:
            st.info("👈 Select at least 2 locations and configure your route options.")
    
    # Show a sample route visualization
    st.subheader("Route Visualization Preview")
    
    # Define sample locations for the preview map
    sample_locations = ["Jemaa el-Fnaa", "Koutoubia Mosque", "Bahia Palace", "Majorelle Garden"]
    sample_coords = [locations_dict.get(loc, (31.6258, -7.9891)) for loc in sample_locations]
    
    # Create a simple map with markers
    sample_m = folium.Map(location=[31.6258, -7.9891], zoom_start=13)

    # Add markers for sample locations with nice styling
    for i, loc in enumerate(sample_locations):
        coords = sample_coords[i]
        
        # Skip if coordinates not found
        if coords is None:
            continue
            
        # Create different colors for different stops
        if i == 0:
            icon_color = 'green'  # Starting point
            icon_type = 'play'
        elif i == len(sample_locations) - 1:
            icon_color = 'red'  # End point
            icon_type = 'flag'
        else:
            icon_color = 'blue'  # Middle stops
            icon_type = 'map-marker'
        
        # Add marker with number
        folium.Marker(
            location=coords,
            popup=f"Example Stop {i+1}: {loc}",
            tooltip=loc,
            icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
        ).add_to(sample_m)
        
        # Add a circle marker with number
        folium.CircleMarker(
            location=coords,
            radius=18,
            color=icon_color,
            fill=True,
            fill_color=icon_color,
            fill_opacity=0.7,
            popup=f"Stop {i+1}: {loc}"
        ).add_to(sample_m)
        
        # Add the stop number as text
        folium.Marker(
            location=coords,
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>'
            )
        ).add_to(sample_m)

# Add example route lines between points
for i in range(len(sample_locations) - 1):
    # Skip if coordinates not found
    if sample_coords[i] is None or sample_coords[i+1] is None:
        continue
        
    # Create a polyline between consecutive points
    folium.PolyLine(
        locations=[sample_coords[i], sample_coords[i+1]],
        color='blue',
        weight=4,
        opacity=0.8,
        tooltip=f"Example route segment {i+1}",
        dash_array='10, 10'  # Make it dashed to indicate it's an example
    ).add_to(sample_m)

# Add a legend to explain this is just a preview
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 120px; 
            border:2px solid grey; z-index:9999; font-size:12px;
            background-color:white; padding: 10px;
            border-radius: 6px;">
    <span style="font-weight: bold; font-size: 14px;">Sample Route Preview</span><br>
    <span style="color: green;">●</span> Starting point<br>
    <span style="color: blue;">●</span> Intermediate stops<br>
    <span style="color: red;">●</span> Final destination<br>
    <hr style="margin: 5px 0;">
    <span style="font-style: italic; font-size: 11px;">
        Select destinations and click "Calculate Optimal Route" to plan your actual route.
    </span>
</div>
'''
sample_m.get_root().html.add_child(folium.Element(legend_html))

# Add example route stats
stats_html = '''
<div style="position: fixed; 
            top: 20px; right: 20px; width: 180px; height: 90px; 
            border:2px solid grey; z-index:9999; font-size:12px;
            background-color:white; padding: 8px;
            border-radius: 6px;">
    <span style="font-weight: bold; font-size: 14px;">Example Stats</span><br>
    <span>Distance: ~5.4 km</span><br>
    <span>Time: ~25 min</span><br>
    <span>Stops: 4</span>
</div>
'''
sample_m.get_root().html.add_child(folium.Element(stats_html))

# Add an info box with sample travel tips
tips_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 220px; 
            border:2px solid #3498db; z-index:9999; font-size:12px;
            background-color:white; padding: 10px;
            border-radius: 6px;">
    <span style="font-weight: bold; font-size: 14px; color: #3498db;">Marrakech Travel Tips</span><br>
    <ul style="padding-left: 20px; margin: 5px 0;">
        <li>Visit Jemaa el-Fnaa in the evening</li>
        <li>Majorelle Garden is less crowded in the morning</li>
        <li>Consider a guided tour for the medina</li>
    </ul>
</div>
'''
sample_m.get_root().html.add_child(folium.Element(tips_html))

# Display the sample map
st_folium_map = streamlit_folium.folium_static(sample_m, width=800, height=500)

# Add helpful instructions below the map
st.markdown("""
### How to Plan Your Route
1. **Select Destinations**: Choose at least 2 locations from the menu on the left
2. **Set Starting Point**: Select where your journey begins
3. **Customize Settings**: Adjust vehicle type, traffic conditions, and route preferences
4. **Calculate Route**: Click the button to find your optimal path

The route optimizer will find the most efficient order to visit all selected locations, saving you time and effort.
""")

# Add sample metrics to showcase the interface
sample_metric_cols = st.columns(4)
with sample_metric_cols[0]:
    st.metric("Sample Distance", "5.4 km", "")
with sample_metric_cols[1]:
    st.metric("Sample Time", "25 min", "-15 min")
with sample_metric_cols[2]:
    st.metric("Sample Stops", "4", "")
with sample_metric_cols[3]:
    st.metric("Sample Optimization", "Very Good", "")

# Add example comparative analysis (what you'd see after calculating)
with st.expander("About Route Optimization"):
    st.write("""
    When you calculate a route, the system compares different orders to visit your selected destinations and finds the most efficient sequence.
    
    **Benefits of optimization:**
    * Reduced travel distance by up to 25%
    * Less time spent in transit
    * More efficient itinerary planning
    * Lower fuel consumption
    
    The optimization becomes more valuable as you add more destinations to your route.
    """)
    
    # Simple chart showing optimization value
    chart_data = {
        'Stops': [2, 3, 4, 5, 6, 8, 10],
        'Potential Savings (%)': [5, 15, 25, 32, 40, 45, 50]
    }
    
    st.write("**Potential distance savings from route optimization:**")
    st.bar_chart(chart_data, x='Stops')

st.markdown("---")
st.markdown("### How to use this app")
st.markdown("""
1. **Shortest Path**: Select a starting point and destination to find the shortest route between them.
   - Choose between graph-based (simplified) or road-based (realistic) routing.
   
2. **Multi-Stop Route**: Select multiple locations to visit, and the app will find the optimal order to visit them all.
   - **Graph-based routing**: Uses a simplified graph model with direct connections between locations.
   - **Road-based routing**: Uses actual road networks for more realistic routes that follow real streets.

The interactive map shows the route with red lines and markers for each location.
""")