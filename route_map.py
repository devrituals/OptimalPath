import folium
from folium.features import DivIcon
import math
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

def calculate_route_distance(coordinates):
    """Calculate the total distance of a route in kilometers"""
    if not coordinates or len(coordinates) < 2:
        return 0
        
    total_distance = 0
    for i in range(len(coordinates)-1):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[i+1]
        total_distance += calculate_haversine_distance(lat1, lon1, lat2, lon2)
        
    return total_distance

def create_route_map(center_coords=(31.6295, -7.9811), zoom_start=13, title="Marrakech Route Map"):
    """Create a folium map for route visualization"""
    m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # Add title
    title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def add_route_to_map(m, route_coords, color='blue', weight=5, opacity=0.8, route_name="Route"):
    """Add a route to the map using PolyLine"""
    # Validate coordinates
    valid_coords = []
    for coord in route_coords:
        if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
            if coord[0] is not None and coord[1] is not None:
                # Ensure coordinates are floats
                try:
                    lat, lon = float(coord[0]), float(coord[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        valid_coords.append((lat, lon))
                except (ValueError, TypeError):
                    continue
    
    # Only add the route if we have at least 2 valid coordinates
    if len(valid_coords) >= 2:
        # Add route line
        folium.PolyLine(
            locations=valid_coords,
            color=color,
            weight=weight,
            opacity=opacity,
            tooltip=route_name
        ).add_to(m)
        
        # Calculate route distance
        distance = calculate_route_distance(valid_coords)
        
        # Add distance label in the middle of the route
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
    
    return m

def add_marker_to_map(m, coords, name="Location", icon=None, number=None):
    """Add a marker to the map"""
    # Validate coordinates
    try:
        lat, lon = float(coords[0]), float(coords[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return m
    except (ValueError, TypeError, IndexError):
        return m
    
    # Add the main marker
    if icon:
        folium.Marker(
            location=(lat, lon),
            popup=name,
            tooltip=name,
            icon=icon
        ).add_to(m)
    else:
        folium.Marker(
            location=(lat, lon),
            popup=name,
            tooltip=name
        ).add_to(m)
    
    # If number is provided, add a numbered marker
    if number is not None:
        # Create a circle marker with the number
        folium.CircleMarker(
            location=(lat, lon),
            radius=18,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.9,
            popup=f"{name} ({number})"
        ).add_to(m)
        
        # Add the number as text
        folium.Marker(
            location=(lat, lon),
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: 12pt; color: white; font-weight: bold; text-align: center;">{number}</div>'
            )
        ).add_to(m)
    
    return m

def fit_map_to_bounds(m, coordinates):
    """Fit the map view to show all the provided coordinates"""
    if not coordinates or len(coordinates) < 2:
        return m
    
    valid_coords = []
    for coord in coordinates:
        if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
            if coord[0] is not None and coord[1] is not None:
                try:
                    lat, lon = float(coord[0]), float(coord[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        valid_coords.append((lat, lon))
                except (ValueError, TypeError):
                    continue
    
    if valid_coords:
        try:
            all_lats = [lat for lat, _ in valid_coords]
            all_lons = [lon for _, lon in valid_coords]
            sw = [min(all_lats), min(all_lons)]
            ne = [max(all_lats), max(all_lons)]
            m.fit_bounds([sw, ne])
        except Exception as e:
            print(f"Error fitting map bounds: {e}")
    
    return m