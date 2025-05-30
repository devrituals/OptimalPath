import folium
from folium.features import DivIcon
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

def calculate_route_distance(coordinates):
    """Calculate the distance of a route in kilometers using the Haversine formula"""
    total_distance = 0
    for i in range(len(coordinates)-1):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[i+1]
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371 * c  # Earth radius in km
        
        total_distance += distance
        
    return total_distance

def create_road_folium_map(center_coords=(31.6295, -7.9811), zoom_start=13, title="Marrakech Map"):
    """Create a folium map for displaying road networks"""
    m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # Add title
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def plot_route_on_map(m, path_coords, color='blue', weight=5, opacity=0.7, route_name="Route"):
    """Plot a route on a folium map using a list of coordinates"""
    # Validate coordinates
    valid_coords = []
    for coord in path_coords:
        if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
            if coord[0] is not None and coord[1] is not None:
                valid_coords.append(coord)
    
    # Only draw if enough valid coordinates
    if len(valid_coords) >= 2:
        # Add path line
        folium.PolyLine(
            locations=valid_coords,
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=route_name
        ).add_to(m)
        
        # Calculate and display distance
        try:
            distance = calculate_route_distance(valid_coords)
            midpoint_idx = len(valid_coords) // 2
            if midpoint_idx > 0:
                # Add distance tooltip
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

def get_route_between_points(start_coords, end_coords, use_cached=True):
    """Simple direct route between points (fallback function)"""
    # Return direct path as fallback
    route_coords = [start_coords, end_coords]
    distance = calculate_route_distance(route_coords)
    return None, route_coords, distance

def plot_route_on_folium(m, route_nodes, G, color='blue', weight=5, opacity=0.7, route_name="Route"):
    """Simplified route plotting (fallback function)"""
    # Just use direct line plotting
    if route_nodes and len(route_nodes) > 0:
        # Extract coordinates
        route_coords = []
        for node in route_nodes:
            try:
                # Add node coordinates if available
                y = G.nodes[node].get('y')
                x = G.nodes[node].get('x')
                if y is not None and x is not None:
                    route_coords.append((y, x))
            except:
                pass
        
        return plot_route_on_map(m, route_coords, color, weight, opacity, route_name)
    return m

def get_road_network_with_cache(place_name="Marrakech, Morocco", force_refresh=False):
    """Return a minimal fallback graph"""
    G = nx.MultiDiGraph()
    
    # Add major landmarks
    locations = {
        'Jemaa el-Fnaa': (31.6258, -7.9891),
        'Koutoubia Mosque': (31.6248, -7.9933),
        'Bahia Palace': (31.6216, -7.9828),
        'Majorelle Garden': (31.6417, -7.9988),
        'Menara Gardens': (31.6147, -8.0162)
    }
    
    # Add nodes
    for name, (lat, lon) in locations.items():
        G.add_node(name, y=lat, x=lon)
    
    # Create edges between all pairs
    nodes = list(G.nodes())
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            # Add edges in both directions
            G.add_edge(n1, n2)
            G.add_edge(n2, n1)
    
    return G