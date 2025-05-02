
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import math
import requests
import io
import csv
import tempfile
import os
import time
import docx
import PyPDF2
from datetime import datetime
import base64
from streamlit.components.v1 import html

# Location tracking services
class LocationTrackingService:
    """Service for tracking user's current location."""
    
    def __init__(self):
        self.tracking_active = False
        self.last_update = None
    
    def start_tracking(self):
        """Start location tracking."""
        self.tracking_active = True
        self.last_update = datetime.now()
        return True
    
    def stop_tracking(self):
        """Stop location tracking."""
        self.tracking_active = False
        return True
    
    def get_tracking_status(self):
        """Get current tracking status."""
        if not self.tracking_active:
            return "Inactive"
        
        if not self.last_update:
            return "Active (waiting for first location)"
        
        time_diff = datetime.now() - self.last_update
        if time_diff.total_seconds() < 60:
            return "Active (updated just now)"
        elif time_diff.total_seconds() < 300:
            return f"Active (updated {int(time_diff.total_seconds() / 60)} min ago)"
        else:
            return f"Active (last update: {self.last_update.strftime('%H:%M:%S')})"

class OptimalPathHandler:
    """Handler for optimal path calculations with current location."""
    
    def __init__(self):
        self.last_update_time = None
        self.current_route = None
        self.next_stop_index = 0
    
    def update_optimal_path(self, stops):
        """Update the optimal path based on current location."""
        if not st.session_state.current_location or not stops:
            return False
        
        # Here we would recalculate the optimal route
        # For now, just update the timestamp
        self.last_update_time = datetime.now()
        st.session_state.last_update_time = self.last_update_time
        return True
    
    def get_distance_to_next_stop(self):
        """Calculate distance and ETA to the next stop."""
        if not st.session_state.current_location or not st.session_state.optimized_route:
            return None, None
        
        if self.next_stop_index >= len(st.session_state.optimized_route):
            return 0, 0  # At final destination
        
        # Get next stop
        next_stop = st.session_state.optimized_route[self.next_stop_index]
        
        # Calculate distance using Haversine formula
        lat1 = st.session_state.current_location['latitude']
        lon1 = st.session_state.current_location['longitude']
        lat2 = next_stop['latitude']
        lon2 = next_stop['longitude']
        
        # Use the haversine distance function from RouteOptimizer
        optimizer = RouteOptimizer()
        distance_m = optimizer._haversine_distance(lat1, lon1, lat2, lon2)
        
        # Convert to km
        distance_km = distance_m / 1000
        
        # Estimate ETA (assuming 50 km/h average speed)
        eta_minutes = (distance_km / 50) * 60
        
        return distance_km, eta_minutes
    # Set page configuration
st.set_page_config(page_title="Route Optimizer", layout="wide")
    
    # Initialize session state
if 'stops' not in st.session_state:
        st.session_state.stops = []
if 'optimized_route' not in st.session_state:
        st.session_state.optimized_route = []
if 'route_paths' not in st.session_state:
        st.session_state.route_paths = []
if 'map_provider' not in st.session_state:
        st.session_state.map_provider = "OpenStreetMap"
if 'distance_units' not in st.session_state:
        st.session_state.distance_units = "Miles"
if 'ors_api_key' not in st.session_state:
        st.session_state.ors_api_key = None
if 'current_location' not in st.session_state:
    st.session_state.current_location = None
if 'tracking_enabled' not in st.session_state:
    st.session_state.tracking_enabled = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'tracking_service' not in st.session_state:
    st.session_state.tracking_service = LocationTrackingService()
if 'path_handler' not in st.session_state:
    st.session_state.path_handler = OptimalPathHandler()
    # =====================================================
    # HELPER FUNCTIONS
    # =====================================================
def get_location():
    """Use browser's geolocation API to get the user's current location with better error handling."""
    # HTML and JavaScript to get user location with debugging
    loc_js = """
    <script>
    // Function to get current position
    function getLocation() {
        console.log("Attempting to get location...");
        if (navigator.geolocation) {
            console.log("Geolocation API is supported");
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    // Success callback
                    console.log("Location acquired successfully:", position.coords);
                    document.getElementById('lat').value = position.coords.latitude;
                    document.getElementById('lng').value = position.coords.longitude;
                    document.getElementById('accuracy').value = position.coords.accuracy;
                    document.getElementById('locationForm').submit();
                },
                function(error) {
                    // Error callback
                    console.error("Error getting location:", error);
                    var errorDiv = document.getElementById('error');
                    var errorMessage = "";
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMessage = "Location permission denied. Please enable location access in your browser settings.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMessage = "Location information is unavailable. Please check if location services are enabled on your device.";
                            break;
                        case error.TIMEOUT:
                            errorMessage = "The request to get location timed out. Please try again or check your internet connection.";
                            break;
                        case error.UNKNOWN_ERROR:
                            errorMessage = "An unknown error occurred while accessing location. Error details: " + error.message;
                            break;
                    }
                    errorDiv.innerHTML = errorMessage;
                    document.getElementById('error-container').style.display = 'block';
                },
                {
                    enableHighAccuracy: true,  // Use GPS if available
                    timeout: 15000,            // Allow more time (15 seconds)
                    maximumAge: 0              // Always get fresh position
                }
            );
        } else {
            console.error("Geolocation is not supported by this browser");
            document.getElementById('error').innerHTML = "Geolocation is not supported by this browser. Please try a different browser like Chrome or Firefox.";
            document.getElementById('error-container').style.display = 'block';
        }
    }
    
    // Check if we're running in a secure context
    if (window.isSecureContext) {
        console.log("Running in secure context, geolocation should work");
    } else {
        console.warn("Not running in secure context! Geolocation API requires HTTPS or localhost");
        document.getElementById('error').innerHTML = "Geolocation requires a secure connection (HTTPS) or localhost. Your current connection appears insecure.";
        document.getElementById('error-container').style.display = 'block';
    }
    
    // Call getLocation on page load
    getLocation();
    </script>
    
    <form id="locationForm" method="POST">
        <input type="hidden" id="lat" name="lat" />
        <input type="hidden" id="lng" name="lng" />
        <input type="hidden" id="accuracy" name="accuracy" />
        <div id="error-container" style="display:none; padding: 10px; background-color: #ffcccc; margin: 10px 0;">
            <p id="error" style="color: red; font-weight: bold;"></p>
        </div>
    </form>
    """
    
    # Show debug info
    st.info("Attempting to access your location. If prompted, please allow location access.")
    
    # Use Streamlit's html component to inject JavaScript
    html(loc_js, height=150)  # Increased height to show errors
    
    # Check for form submission with location data
    query_params = st.experimental_get_query_params()
    if query_params:
        st.write("Received query parameters:", query_params)  # Debug info
        if 'lat' in query_params and 'lng' in query_params:
            try:
                lat = float(query_params['lat'][0])
                lng = float(query_params['lng'][0])
                accuracy = float(query_params['accuracy'][0]) if 'accuracy' in query_params else None
                
                # Update session state with current location
                st.session_state.current_location = {
                    'latitude': lat,
                    'longitude': lng,
                    'accuracy': accuracy,
                    'timestamp': datetime.now()
                }
                
                # Update last update time
                st.session_state.last_update_time = datetime.now()
                
                # Clear query parameters to avoid reloading issues
                st.experimental_set_query_params()
                
                return st.session_state.current_location
            except (ValueError, KeyError) as e:
                st.error(f"Error processing location data: {e}")
                st.write("Failed to process:", query_params)  # Debug info
                st.experimental_set_query_params()
    
    return st.session_state.current_location

def setup_location_tracking():
    """Set up JavaScript for continuous location tracking with fallbacks."""
    tracking_js = """
    <script>
    let watchId = null;
    
    // Simple fallback for older browsers
    function getLocationFallback() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    updateLocation(position.coords.latitude, position.coords.longitude, position.coords.accuracy);
                },
                function(error) {
                    document.getElementById('tracking-error').innerHTML = "Error: " + error.message;
                    document.getElementById('tracking-error').style.display = 'block';
                }
            );
        }
    }
    
    function startTracking() {
        if (navigator.geolocation) {
            try {
                // Start watching position
                watchId = navigator.geolocation.watchPosition(
                    function(position) {
                        // Success callback
                        updateLocation(position.coords.latitude, position.coords.longitude, position.coords.accuracy);
                    },
                    function(error) {
                        // Error callback
                        console.error("Error getting location: ", error);
                        document.getElementById('tracking-error').innerHTML = "Error: " + error.message;
                        document.getElementById('tracking-error').style.display = 'block';
                        
                        // Try fallback on error
                        setTimeout(getLocationFallback, 2000);
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 15000,
                        maximumAge: 0
                    }
                );
                document.getElementById('tracking-status').innerHTML = "Tracking: Active";
                document.getElementById('tracking-status').style.color = 'green';
            } catch (e) {
                console.error("Exception starting tracking: ", e);
                document.getElementById('tracking-error').innerHTML = "Error starting tracking: " + e.message;
                document.getElementById('tracking-error').style.display = 'block';
            }
        } else {
            document.getElementById('tracking-error').innerHTML = "Geolocation is not supported by this browser.";
            document.getElementById('tracking-error').style.display = 'block';
        }
    }
    
    function updateLocation(lat, lng, accuracy) {
        // Send location to Streamlit
        const data = {
            lat: lat,
            lng: lng,
            accuracy: accuracy,
            timestamp: new Date().toISOString()
        };
        
        // Use Streamlit's setComponentValue when available
        if (window.parent && window.parent.postMessage) {
            const message = {
                type: "streamlit:setComponentValue",
                value: data
            };
            window.parent.postMessage(message, "*");
        }
        
        // Also update UI elements
        document.getElementById('current-lat').textContent = lat.toFixed(6);
        document.getElementById('current-lng').textContent = lng.toFixed(6);
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    }
    
    function stopTracking() {
        if (watchId !== null) {
            navigator.geolocation.clearWatch(watchId);
            watchId = null;
            document.getElementById('tracking-status').innerHTML = "Tracking: Inactive";
            document.getElementById('tracking-status').style.color = 'red';
        }
    }
    
    // Start tracking when the page loads if enabled
    if (document.getElementById('tracking-enabled').value === 'true') {
        startTracking();
    }
    </script>
    
    <input type="hidden" id="tracking-enabled" value="${'true' if st.session_state.tracking_enabled else 'false'}" />
    
    <div id="tracking-container">
        <div id="tracking-status" style="color: ${('green' if st.session_state.tracking_enabled else 'red')}">
            Tracking: ${('Active' if st.session_state.tracking_enabled else 'Inactive')}
        </div>
        <div id="location-display">
            <div>Current Position: <span id="current-lat">-</span>, <span id="current-lng">-</span></div>
            <div>Last Update: <span id="last-update">-</span></div>
        </div>
        <div id="tracking-error" style="display:none; color: red;"></div>
    </div>
    """
    
    # Use Streamlit's html component to inject JavaScript
    html(tracking_js, height=150)

def create_map(route, route_paths=None, map_provider="OpenStreetMap"):
    """
    Create a folium map with the route.
    
    Args:
        route: List of location dictionaries
        route_paths: List of paths between consecutive stops, each a list of [lat, lng] coordinates
        map_provider: Map tile provider to use
    """
    # Handle the case where map_provider is passed as the second argument (backward compatibility)
    if isinstance(route_paths, str):
        map_provider = route_paths
        route_paths = None
    
    # Determine map center (average of all points)
    lats = [stop['latitude'] for stop in route if 'latitude' in stop]
    lngs = [stop['longitude'] for stop in route if 'longitude' in stop]
    
    if not lats or not lngs:
        st.error("No valid coordinates to display on the map.")
        return folium.Map(location=[40.0, -95.0], zoom_start=4)
    
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)
    
    # Create map with selected provider
    tiles = "OpenStreetMap"
    if map_provider == "CartoDB":
        tiles = "CartoDB positron"
    elif map_provider == "Stamen":
        tiles = "Stamen Terrain"
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, tiles=tiles)
    
    # Track which stops have valid coordinates to ensure proper numbering
    valid_stops = []
    for stop in route:
        if 'latitude' in stop and 'longitude' in stop:
            valid_stops.append(stop)
    
    # Add stops as markers with numbers
    for i, stop in enumerate(valid_stops):
        popup_text = f"{stop['name']}<br>{stop['address']}"
        
        # Set marker style and text
        if i == 0:
            # First stop is always the starting point
            bg_color = "#38761d"  # green for start
            text_color = "white"
            marker_text = "S"  # S for Start
        else:
            # Regular numbered stops - number sequentially
            bg_color = "white"
            text_color = "black"
            # Use i instead of i+1 to ensure consecutive numbering
            marker_text = str(i)  # This will give 1, 2, 3...
        
        # Create HTML for the marker
        icon_html = f'''
            <div style="
                font-size: 12pt; 
                background-color: {bg_color}; 
                color: {text_color};
                border: 2px solid black; 
                border-radius: 50%; 
                width: 30px; 
                height: 30px; 
                line-height: 28px; 
                text-align: center;
                box-shadow: 0 0 3px rgba(0,0,0,0.4);
            "><b>{marker_text}</b></div>
        '''
        
        # Create the marker with DivIcon
        folium.Marker(
            [stop['latitude'], stop['longitude']], 
            popup=popup_text,
            icon=folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=icon_html
            )
        ).add_to(m)
    
    # Add route line
    if route_paths and len(route_paths) > 0:
        # Use the detailed road paths
        for i, path in enumerate(route_paths):
            if path and isinstance(path, list) and len(path) > 1:
                try:
                    # Validate that path contains valid coordinates
                    for point in path:
                        if not isinstance(point, list) or len(point) < 2:
                            continue
                        float(point[0])  # Test if convertible to float
                        float(point[1])
                    
                    # Create a color gradient from blue to red
                    color = _get_color_for_segment(i, len(route_paths))
                    
                    folium.PolyLine(
                        path,
                        color=color,
                        weight=4,
                        opacity=0.8
                    ).add_to(m)
                except (ValueError, TypeError) as e:
                    st.warning(f"Invalid coordinates in path: {e}")
    elif st.session_state.optimized_route:
        # Fallback to straight lines if no paths provided
        route_points = [[stop['latitude'], stop['longitude']] 
                      for stop in st.session_state.optimized_route 
                      if 'latitude' in stop and 'longitude' in stop]
        
        if route_points and len(route_points) > 1:
            folium.PolyLine(
                route_points,
                color='blue',
                weight=3,
                opacity=0.7,
                dash_array='5'  # Use dashed line to indicate this is not a road path
            ).add_to(m)
    
    return m 

def _get_color_for_segment(index, total):
    """Generate a color gradient from blue to red based on segment position."""
    if total <= 1:
        return 'blue'
        
    # Create gradient using HSL (hue, saturation, lightness)
    # Blue (240°) to Red (0°)
    hue = 240 - (240 * index / (total - 1))
    return f'hsl({hue}, 100%, 50%)'

def geocode_stops():
        """Geocode all stops that don't have coordinates."""
        stops_to_geocode = [s for s in st.session_state.stops if 'latitude' not in s or 'longitude' not in s]
        
        if not stops_to_geocode:
            st.success("All stops already have coordinates.")
            return
        
        with st.spinner(f"Geocoding {len(stops_to_geocode)} stops..."): 
            progress_bar = st.progress(0.0)
            
            geocoder = GeocodingService()
            geocoded_stops = geocoder.batch_geocode(stops_to_geocode, progress_bar)
            
            # Update the stops with geocoded data
            geocoded_count = 0
            for i, stop in enumerate(st.session_state.stops):
                if 'latitude' not in stop or 'longitude' not in stop:
                    # Find matching geocoded stop
                    match = next((s for s in geocoded_stops if s.get('name') == stop.get('name') and
                                 s.get('address') == stop.get('address')), None)
                    
                    if match and 'latitude' in match and 'longitude' in match:
                        st.session_state.stops[i]['latitude'] = match['latitude']
                        st.session_state.stops[i]['longitude'] = match['longitude']
                        geocoded_count += 1
            
        if geocoded_count > 0:
            st.success(f"Successfully geocoded {geocoded_count} stops.")
        else:
            st.error("Could not geocode any stops. Please check addresses.")
    
def generate_report():
        """Generate a report of the route."""
        route = st.session_state.optimized_route if st.session_state.optimized_route else st.session_state.stops
        
        report = "# Route Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total Stops: {len(route)}\n\n"
        
        # Calculate total distance if we have coordinates
        if all('latitude' in stop and 'longitude' in stop for stop in route):
            total_distance = 0
            optimizer = RouteOptimizer()
            
            for i in range(len(route) - 1):
                distance = optimizer._haversine_distance(
                    route[i]['latitude'], route[i]['longitude'],
                    route[i+1]['latitude'], route[i+1]['longitude']
                )
                
                # Convert to miles or km based on settings
                if st.session_state.distance_units == "Miles":
                    distance = distance / 1609.34  # meters to miles
                    distance_unit = "mi"
                else:
                    distance = distance / 1000  # meters to km
                    distance_unit = "km"
                    
                total_distance += distance
            
            report += f"Total Distance: {total_distance:.2f} {distance_unit}\n\n"
        
        # Add route details
        report += "## Route Details\n\n"
        
        for i, stop in enumerate(route):
            report += f"### Stop {i+1}: {stop.get('name', 'Unnamed')}\n"
            report += f"Address: {stop.get('address', 'No address')}\n"
            
            if 'latitude' in stop and 'longitude' in stop:
                report += f"Coordinates: {stop.get('latitude')}, {stop.get('longitude')}\n"
            
            # Add distance to next stop if we have coordinates and it's not the last stop
            if i < len(route) - 1 and 'latitude' in stop and 'longitude' in stop and 'latitude' in route[i+1] and 'longitude' in route[i+1]:
                optimizer = RouteOptimizer()
                distance = optimizer._haversine_distance(
                    stop['latitude'], stop['longitude'],
                    route[i+1]['latitude'], route[i+1]['longitude']
                )
                
                # Convert to miles or km based on settings
                if st.session_state.distance_units == "Miles":
                    distance = distance / 1609.34  # meters to miles
                    distance_unit = "mi"
                else:
                    distance = distance / 1000  # meters to km
                    distance_unit = "km"
                
                report += f"Distance to next stop: {distance:.2f} {distance_unit}\n"
            
            report += "\n"
        
        # Store the report in session state and display it
        st.session_state.current_report = report
        st.markdown(report)
        
        return report
    
    # Document Parser Class
class LocationParser:
        """
        Class to parse location data from various document formats.
        Supports CSV, TXT, PDF, and DOCX files.
        """
       
        def parse_document(self, uploaded_file):
            """Parse location data from an uploaded file."""
            file_name = uploaded_file.name
            _, file_extension = os.path.splitext(file_name)
            file_extension = file_extension.lower()
            
            if file_extension == '.csv':
                return self._parse_csv(uploaded_file)
            elif file_extension == '.txt':
                return self._parse_txt(uploaded_file)
            elif file_extension == '.pdf':
                return self._parse_pdf(uploaded_file)
            elif file_extension in ['.docx', '.doc']:
                return self._parse_docx(uploaded_file)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                return self._parse_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return []
        
        def _parse_csv(self, uploaded_file):
            """Parse location data from CSV file."""
            locations = []
            try:
                df = pd.read_csv(uploaded_file)
                for _, row in df.iterrows():
                    location = self._process_dataframe_row(row)
                    if location:
                        locations.append(location)
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
            return locations
        
        def _parse_excel(self, uploaded_file):
            """Parse location data from Excel file."""
            locations = []
            try:
                df = pd.read_excel(uploaded_file)
                for _, row in df.iterrows():
                    location = self._process_dataframe_row(row)
                    if location:
                        locations.append(location)
            except Exception as e:
                st.error(f"Error parsing Excel file: {e}")
            return locations
        
        def _process_dataframe_row(self, row):
            """Process a pandas dataframe row to get location data."""
            location = {}
            # Convert to dictionary for consistent processing
            row_dict = row.to_dict()
            return self._process_location_data(row_dict)
        
        def _parse_txt(self, uploaded_file):
            """Parse location data from TXT file."""
            locations = []
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if ',' in line:
                    parts = [part.strip() for part in line.split(',')]
                else:
                    parts = [part.strip() for part in line.split('\t')]
                
                if len(parts) >= 2:  # At minimum, we need name and address
                    location = {
                        'name': parts[0],
                        'address': parts[1],
                    }
                    
                    # Add coordinates if available
                    if len(parts) >= 4:
                        try:
                            location['latitude'] = float(parts[2])
                            location['longitude'] = float(parts[3])
                        except ValueError:
                            # If coordinates are not valid, we'll geocode later
                            pass
                    
                    locations.append(location)
            
            return locations
        
        def _parse_pdf(self, uploaded_file):
            """Parse location data from PDF file."""
            locations = []
            
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(uploaded_file.getvalue())
                temp_path = temp.name
            
            try:
                with open(temp_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                # Process extracted text line by line
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    if ',' in line and len(line.split(',')) >= 2:
                        parts = [part.strip() for part in line.split(',')]
                        location = self._process_location_parts(parts)
                        if location:
                            locations.append(location)
            except Exception as e:
                st.error(f"Error parsing PDF: {e}")
            finally:
                os.unlink(temp_path)  # Clean up the temp file
            
            return locations
        
        def _parse_docx(self, uploaded_file):
            """Parse location data from DOCX file."""
            locations = []
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp:
                temp.write(uploaded_file.getvalue())
                temp_path = temp.name
            
            try:
                doc = docx.Document(temp_path)
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if not text:  # Skip empty paragraphs
                        continue
                        
                    if ',' in text:
                        parts = [part.strip() for part in text.split(',')]
                        location = self._process_location_parts(parts)
                        if location:
                            locations.append(location)
            except Exception as e:
                st.error(f"Error parsing DOCX: {e}")
            finally:
                os.unlink(temp_path)  # Clean up the temp file
            
            return locations
        
        def _process_location_data(self, data):
            """Process a location data dictionary to standardize keys and values."""
            location = {}
            
            # Map common column names to standard keys
            name_keys = ['name', 'location name', 'location', 'place', 'stop name']
            address_keys = ['address', 'location address', 'full address', 'stop address']
            lat_keys = ['latitude', 'lat', 'y']
            lng_keys = ['longitude', 'lng', 'long', 'x']
            
            # Find the name
            for key in name_keys:
                if key.lower() in [k.lower() for k in data.keys()]:
                    key = next(k for k in data.keys() if k.lower() == key.lower())
                    if data[key]:
                        location['name'] = data[key]
                        break
            
            # Find the address
            for key in address_keys:
                if key.lower() in [k.lower() for k in data.keys()]:
                    key = next(k for k in data.keys() if k.lower() == key.lower())
                    if data[key]:
                        location['address'] = data[key]
                        break
            
            # Find coordinates
            for key in lat_keys:
                if key.lower() in [k.lower() for k in data.keys()]:
                    key = next(k for k in data.keys() if k.lower() == key.lower())
                    try:
                        if data[key]:
                            location['latitude'] = float(data[key])
                    except (ValueError, TypeError):
                        pass
            
            for key in lng_keys:
                if key.lower() in [k.lower() for k in data.keys()]:
                    key = next(k for k in data.keys() if k.lower() == key.lower())
                    try:
                        if data[key]:
                            location['longitude'] = float(data[key])
                    except (ValueError, TypeError):
                        pass
            
            # Require at least a name or address
            if ('name' in location or 'address' in location):
                # If name is missing, use address as name or a placeholder
                if 'name' not in location:
                    location['name'] = location.get('address', 'Location')
                # If address is missing, use name as address
                if 'address' not in location:
                    location['address'] = location.get('name', '')
                return location
            return None
        
        def _process_location_parts(self, parts):
            """Process a list of strings that may contain location data."""
            if len(parts) < 2:
                return None
            
            location = {
                'name': parts[0],
                'address': parts[1]
            }
            
            # Try to extract coordinates if available
            if len(parts) >= 4:
                try:
                    location['latitude'] = float(parts[2])
                    location['longitude'] = float(parts[3])
                except ValueError:
                    pass
            
            return location
    
    # Geocoding Service
class GeocodingService:
        """Service for geocoding addresses to obtain coordinates."""
        
        def geocode_location(self, location):
            """
            Geocode a location if it doesn't have coordinates.
            
            Args:
                location: Location dictionary with 'address' and optionally 'name'
                
            Returns:
                Location with added 'latitude' and 'longitude' if geocoding was successful
            """
            # Skip if we already have coordinates
            if 'latitude' in location and 'longitude' in location:
                return location
            
            # Can't geocode without an address
            if 'address' not in location or not location['address']:
                return location
            
            address = location['address']
            
            # Try Nominatim (OpenStreetMap) API - no API key required
            coordinates = self._geocode_with_nominatim(address)
            
            if coordinates:
                location['latitude'], location['longitude'] = coordinates
            
            return location
        
        def _geocode_with_nominatim(self, address):
            """Geocode using Nominatim (OpenStreetMap) API."""
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': address,
                    'format': 'json',
                    'limit': 1
                }
                headers = {
                    'User-Agent': 'StreamlitLocationOptimizer/1.0'  # Required by Nominatim
                }
                response = requests.get(url, params=params, headers=headers)
                data = response.json()
                
                if data and len(data) > 0:
                    return float(data[0]['lat']), float(data[0]['lon'])
            except Exception as e:
                st.error(f"Geocoding error: {e}")
            
            return None
        
        def batch_geocode(self, locations, progress_bar=None):
            """Geocode multiple locations with optional progress tracking."""
            total = len(locations)
            geocoded = []
            
            for i, location in enumerate(locations):
                geocoded.append(self.geocode_location(location))
                if progress_bar:
                    progress_bar.progress((i + 1) / total)
                # Be nice to the geocoding service
                time.sleep(1)
            
            return geocoded
    
    # Route Optimizer
class RouteOptimizer:
    """Service for finding the optimal route through a set of locations."""
    
    def __init__(self, api_key=None):
        """Initialize with optional API key for routing services."""
        self.api_key = api_key
    
    def optimize_route(self, locations, start_index=0, use_real_roads=True, start_at_current_location=False, cost_type="distance"):
        """
        Find the optimal route through all locations.
        
        Args:
            locations: List of location dictionaries with 'latitude' and 'longitude'
            start_index: Index of the starting location
            use_real_roads: Whether to use real road routing or just optimize stop order
            start_at_current_location: Whether to use current location as starting point
            cost_type: "distance" or "time" for optimization criteria
            
        Returns:
            Tuple of (optimized_route, route_paths)
        """
        if not locations:
            return [], []
        
        # Ensure we have coordinates for all locations
        if not all('latitude' in loc and 'longitude' in loc for loc in locations):
            st.error("Not all locations have coordinates")
            return locations, []
        
        # Calculate the distance matrix
        distance_matrix = self._calculate_distance_matrix(locations)
        
        # Apply the nearest neighbor algorithm
        route_indices = self._nearest_neighbor_tsp(distance_matrix, start_index)
        
        # Reorder locations based on the route
        optimized_route = [locations[i] for i in route_indices]
        
        # If real roads requested, get the actual route paths
        route_paths = []
        if use_real_roads and len(optimized_route) > 1:
            with st.spinner(f"Calculating routes between {len(optimized_route)-1} stops..."):
                progress_bar = st.progress(0.0)
                route_paths = []
                
                for i in range(len(optimized_route) - 1):
                    start = optimized_route[i]
                    end = optimized_route[i + 1]
                    
                    path = self._get_route_between_points(
                        start['longitude'], start['latitude'], 
                        end['longitude'], end['latitude'],
                        self.api_key
                    )
                    
                    if path:
                        route_paths.append(path)
                    else:
                        # Fallback to straight line if routing fails
                        route_paths.append([
                            [start['latitude'], start['longitude']],
                            [end['latitude'], end['longitude']]
                        ])
                    
                    # Update progress
                    progress_bar.progress((i + 1) / (len(optimized_route) - 1))
                
                progress_bar.empty()
        
        return optimized_route, route_paths
    
    def _calculate_distance_matrix(self, locations):
        """Calculate the distance matrix between all locations."""
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distance = self._haversine_distance(
                    locations[i]['latitude'], locations[i]['longitude'],
                    locations[j]['latitude'], locations[j]['longitude']
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on the Earth."""
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        r = 6371000
        return c * r
    
    def _nearest_neighbor_tsp(self, distance_matrix, start_index):
        """Solve the TSP using the Nearest Neighbor heuristic."""
        n = distance_matrix.shape[0]
        visited = [False] * n
        route = [start_index]
        visited[start_index] = True
        
        current = start_index
        for _ in range(n - 1):
            # Find the nearest unvisited location
            nearest = None
            min_distance = float('inf')
            
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < min_distance:
                    min_distance = distance_matrix[current, j]
                    nearest = j
            
            if nearest is not None:
                route.append(nearest)
                visited[nearest] = True
                current = nearest
        
        return route
    
    def _get_real_road_paths(self, route):
        """
        Get actual road paths between consecutive stops using OpenRouteService API.
        
        Args:
            route: List of location dictionaries ordered by the optimized route
            
        Returns:
            List of paths between consecutive stops, each path is a list of [lat, lng] coordinates
        """
        # This method is kept for backward compatibility
        # The actual implementation is now in optimize_route
        paths = []
        
        if len(route) < 2:
            return paths
        
        # Use OpenRouteService for routing if API key provided, otherwise fallback to OSRM
        api_key = self.api_key if self.api_key else st.session_state.get('ors_api_key', None)
        
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            
            path = self._get_route_between_points(
                start['longitude'], start['latitude'], 
                end['longitude'], end['latitude'],
                api_key
            )
            
            if path:
                paths.append(path)
            else:
                # Fallback to straight line if routing fails
                paths.append([
                    [start['latitude'], start['longitude']],
                    [end['latitude'], end['longitude']]
                ])
        
        return paths
    
    def _get_route_between_points(self, lon1, lat1, lon2, lat2, api_key=None):
        """
        Get the actual road route between two points using a routing API.
        
        Args:
            lon1, lat1: Coordinates of the starting point
            lon2, lat2: Coordinates of the ending point
            api_key: API key for OpenRouteService
            
        Returns:
            List of [lat, lng] coordinates representing the path
        """
        # Try OpenRouteService first if API key is available
        if api_key:
            try:
                url = "https://api.openrouteservice.org/v2/directions/driving-car"
                headers = {
                    'Authorization': api_key,
                    'Content-Type': 'application/json'
                }
                body = {
                    "coordinates": [[lon1, lat1], [lon2, lat2]],
                    "format": "geojson"
                }
                
                response = requests.post(url, json=body, headers=headers)
                
                if response.status_code == 200:
                    route_data = response.json()
                    # Extract coordinates from GeoJSON
                    coordinates = route_data['features'][0]['geometry']['coordinates']
                    # Convert from [lng, lat] to [lat, lng] for folium
                    return [[coord[1], coord[0]] for coord in coordinates]
                elif response.status_code == 401:
                    st.warning("Invalid API key for OpenRouteService. Falling back to OSRM.")
                elif response.status_code == 429:
                    st.warning("OpenRouteService rate limit exceeded. Falling back to OSRM.")
                else:
                    st.warning(f"OpenRouteService error (status {response.status_code}). Falling back to OSRM.")
            except Exception as e:
                st.warning(f"Error with OpenRouteService API: {e}. Falling back to OSRM.")
        
        # Fallback to OSRM (Open Source Routing Machine) which doesn't require API key
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                route_data = response.json()
                if route_data['code'] == 'Ok':
                    # Extract coordinates from GeoJSON
                    coordinates = route_data['routes'][0]['geometry']['coordinates']
                    # Convert from [lng, lat] to [lat, lng] for folium
                    return [[coord[1], coord[0]] for coord in coordinates]
                else:
                    st.warning(f"OSRM routing error: {route_data.get('message', 'Unknown error')}. Using straight line.")
            else:
                st.warning(f"OSRM API error (status {response.status_code}). Using straight line.")
        except Exception as e:
            st.warning(f"Error with OSRM API: {e}. Using straight line instead.")
        
        return None
    
    # =====================================================
    # STREAMLIT APP UI
    # =====================================================
    
    # App header
st.title("Route Optimizer")
st.write("Upload locations and find the optimal route")
    
    # Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stop Manager", "Map View", "Reports", "Settings", "Live Tracking"])

# Live Tracking tab
with tab5:
    st.header("Live Location Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tracking_enabled = st.checkbox("Enable Live Tracking", 
                                     value=st.session_state.tracking_enabled,
                                     help="Use your current location to optimize the route in real-time")
        
        # Handle tracking state changes
        if tracking_enabled != st.session_state.tracking_enabled:
            st.session_state.tracking_enabled = tracking_enabled
            
            if tracking_enabled:
                # Start tracking
                st.session_state.tracking_service.start_tracking()
                st.success("Live tracking enabled! Your current location will be used to optimize the route.")
            else:
                # Stop tracking
                st.session_state.tracking_service.stop_tracking()
                st.info("Live tracking disabled.")
    
    with col2:
        if st.button("Get Current Location"):
            with st.spinner("Getting your location..."):
                location = get_location()
                if location:
                    st.success("Location acquired!")
                    st.session_state.last_update_time = datetime.now()
                else:
                    st.error("Could not get your location. Please check browser permissions.")
    
    # Show tracking status if enabled
    if tracking_enabled:
        st.info(f"Tracking status: {st.session_state.tracking_service.get_tracking_status()}")
        
        # Get current location
        current_loc = st.session_state.get('current_location')
        if current_loc:
            st.metric("Latitude", f"{current_loc['latitude']:.6f}")
            st.metric("Longitude", f"{current_loc['longitude']:.6f}")
            
            if 'accuracy' in current_loc:
                st.metric("Accuracy", f"{current_loc['accuracy']:.1f} meters")
            
            if 'timestamp' in current_loc:
                st.text(f"Last updated: {current_loc['timestamp'].strftime('%H:%M:%S')}")
            
            # Show distance to next stop if available
            if st.session_state.optimized_route:
                path_handler = st.session_state.path_handler
                distance, eta = path_handler.get_distance_to_next_stop()
                if distance is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Distance to next stop", f"{distance:.2f} km")
                    with col2:
                        st.metric("ETA", f"{eta:.1f} minutes")
                
                # Show current route
                st.subheader("Current Route")
                if st.session_state.optimized_route:
                    # Create a simple table of stops
                    preview_data = []
                    for i, stop in enumerate(st.session_state.optimized_route):
                        preview_data.append({
                            "Stop #": i+1,
                            "Name": stop.get('name', 'Unnamed'),
                            "Address": stop.get('address', 'No address')
                        })
                    
                    st.table(pd.DataFrame(preview_data))
                    
                    # Add a mini-map with current location
                    st.subheader("Live Tracking Map")
                    
                    # Add current location to the map
                    if current_loc:
                        current_pos = {
                            'name': 'Current Position',
                            'address': 'My Location',
                            'latitude': current_loc['latitude'],
                            'longitude': current_loc['longitude']
                        }
                        
                        # Create map with current location and route
                        route_with_current = [current_pos] + st.session_state.optimized_route
                        mini_map = create_map(route_with_current, st.session_state.route_paths, st.session_state.map_provider)
                        folium_static(mini_map, width=800, height=500)
                    
                    # Update route button
                    if st.button("Update Route Based on Current Location"):
                        with st.spinner("Updating route..."):
                            if st.session_state.path_handler.update_optimal_path(st.session_state.stops):
                                st.success("Route updated successfully!")
                                st.rerun()
                            else:
                                st.warning("No update needed or current location not available.")
        else:
            st.warning("Waiting for location data... Please make sure you've allowed location access in your browser.")
            
            # Set up the location tracking JavaScript
            setup_location_tracking()
    else:
        st.info("Enable live tracking to see your current location and optimize routes in real-time.")

    # Settings tab
with tab4:
        st.header("Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            map_provider = st.selectbox(
                "Map Provider",
                ["OpenStreetMap", "CartoDB", "Stamen"],
                index=0
            )
            if map_provider != st.session_state.map_provider:
                st.session_state.map_provider = map_provider
        
        with col2:
            distance_units = st.selectbox(
                "Distance Units",
                ["Miles", "Kilometers"],
                index=0
            )
            if distance_units != st.session_state.distance_units:
                st.session_state.distance_units = distance_units
        
        st.subheader("Routing API Settings")
        st.info("To enable real road routing, you need an OpenRouteService API key. Get one for free at https://openrouteservice.org/")
        
        ors_api_key = st.text_input("OpenRouteService API Key", 
                                   value=st.session_state.ors_api_key if st.session_state.ors_api_key else "",
                                   type="password",
                                   help="Required for real road routing")
        
        if st.button("Save Settings"):
            st.session_state.ors_api_key = ors_api_key if ors_api_key else None
            st.success("Settings saved!")
    
    # Stop Manager tab
with tab1:
        st.header("Manage Stops")
        
        # Add Stop manually
        with st.expander("Add Stop Manually"):
            with st.form("add_stop_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Name")
                    lat = st.text_input("Latitude (optional)")
                with col2:
                    address = st.text_input("Address")
                    lng = st.text_input("Longitude (optional)")
                
                submitted = st.form_submit_button("Add Stop")
                if submitted:
                    if not name or not address:
                        st.error("Please provide both a name and address.")
                    else:
                        stop = {
                            "name": name,
                            "address": address
                        }
                        
                        # Add coordinates if provided
                        if lat and lng:
                            try:
                                stop["latitude"] = float(lat)
                                stop["longitude"] = float(lng)
                            except ValueError:
                                st.error("Coordinates must be numeric values.")
                                stop = None
                        
                        if stop:
                            st.session_state.stops.append(stop)
                            st.session_state.optimized_route = []  # Invalidate optimized route
                            st.success(f"Added stop: {name}")
        
        # Import stops from document
        with st.expander("Import Stops from Document"):
            uploaded_file = st.file_uploader("Choose a file", 
                                            type=['csv', 'txt', 'pdf', 'docx', 'xlsx', 'xls'])
            
            if uploaded_file is not None:
                if st.button("Import Stops"):
                    with st.spinner("Parsing document..."):
                        parser = LocationParser()
                        locations = parser.parse_document(uploaded_file)
                        
                        if not locations:
                            st.error("No valid locations found in the document.")
                        else:
                            if st.session_state.stops:
                                replace = st.radio(
                                    f"Found {len(locations)} locations. How would you like to proceed?",
                                    ["Replace existing stops", "Append to existing stops"]
                                )
                                if replace == "Replace existing stops":
                                    st.session_state.stops = []
                            
                            # Add new locations to stops
                            st.session_state.stops.extend(locations)
                            st.session_state.optimized_route = []  # Invalidate optimized route
                            
                            # Check if we need to geocode
                            missing_coords = [s for s in locations if 'latitude' not in s or 'longitude' not in s]
                            if missing_coords:
                                if st.button("Geocode missing coordinates"):
                                    geocode_stops()
                            
                            st.success(f"Imported {len(locations)} stops")
        
        # Display stops in a table
        if st.session_state.stops:
            st.subheader("Current Stops")
            
            # Enable editing of current stops
            edited_df = st.data_editor(
                pd.DataFrame(st.session_state.stops),
                key="stops_editor",
                hide_index=True,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "name": "Name",
                    "address": "Address",
                    "latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                    "longitude": st.column_config.NumberColumn("Longitude", format="%.6f")
                }
            )
            
            # Update stops if edited
            if not edited_df.equals(pd.DataFrame(st.session_state.stops)):
                st.session_state.stops = edited_df.to_dict('records')
                st.session_state.optimized_route = []  # Invalidate optimized route
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Geocode Missing Coordinates"):
                    geocode_stops()
            
            with col2:
                if st.button("Clear All Stops"):
                    st.session_state.stops = []
                    st.session_state.optimized_route = []
                    st.rerun()
        
            # Optimize route
            if len(st.session_state.stops) >= 2:
                st.subheader("Route Optimization")
                
                # Check if we have coordinates for all stops
                missing_coords = [s for s in st.session_state.stops if 'latitude' not in s or 'longitude' not in s]
                if missing_coords:
                    st.warning(f"{len(missing_coords)} stops are missing coordinates. Please geocode them first.")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_options = ["First stop in list"]
                        
                        # Add current location option if tracking is enabled
                        if st.session_state.tracking_enabled and st.session_state.current_location:
                            start_options.insert(0, "Current Location")
                        
                        # Add all stops
                        start_options.extend([stop['name'] for stop in st.session_state.stops])
                        
                        start_option = st.selectbox(
                            "Choose starting point",
                            start_options
                        )
                        
                        use_real_roads = st.checkbox("Follow real road network", value=True, 
                                                   help="When enabled, routes will follow actual roads instead of straight lines.")
                        
                        optimization_criteria = st.radio(
                            "Optimize for",
                            ["Distance", "Time"],
                            horizontal=True
                        )
                        
                        if use_real_roads and not st.session_state.ors_api_key:
                            st.warning("Real road routing requires an OpenRouteService API key. Please add one in Settings.")
                    
                    with col2:
                        if st.button("Optimize Route"):
                            with st.spinner("Optimizing route..."):
                                # Handle current location as starting point
                                use_current_location = start_option == "Current Location"
                                
                                # Determine starting index for regular stops
                                if start_option == "First stop in list":
                                    start_index = 0
                                elif not use_current_location:
                                    names = [stop['name'] for stop in st.session_state.stops]
                                    start_index = names.index(start_option)
                                else:
                                    start_index = 0  # Will be ignored when using current location
                                
                                # Get API key from session state
                                api_key = st.session_state.ors_api_key
                                
                                # If real roads selected but no API key, show warning
                                if use_real_roads and not api_key:
                                    st.warning("No API key available. Using straight-line distances instead.")
                                    use_real_roads = False
                                
                                optimizer = RouteOptimizer(api_key=api_key)
                                
                                # Handle current location if tracking is enabled
                                if use_current_location and st.session_state.current_location:
                                    # Create location entry for current position
                                    current_pos = {
                                        'name': 'Current Position',
                                        'address': 'My Location',
                                        'latitude': st.session_state.current_location['latitude'],
                                        'longitude': st.session_state.current_location['longitude'],
                                        'is_current_location': True
                                    }
                                    
                                    # Insert current position as first stop
                                    stops_with_current = [current_pos] + st.session_state.stops
                                    
                                    optimized_route, route_paths = optimizer.optimize_route(
                                        stops_with_current,
                                        start_index=0,  # Start at current location (index 0)
                                        use_real_roads=use_real_roads,
                                        start_at_current_location=True,
                                        cost_type="distance" if optimization_criteria == "Distance" else "time"
                                    )
                                else:
                                    # Regular optimization without current location
                                    optimized_route, route_paths = optimizer.optimize_route(
                                        st.session_state.stops, 
                                        start_index,
                                        use_real_roads=use_real_roads,
                                        cost_type="distance" if optimization_criteria == "Distance" else "time"
                                    )
                                
                                st.session_state.optimized_route = optimized_route
                                st.session_state.route_paths = route_paths
                                
                                if optimized_route:
                                    st.success("Route optimized successfully!")
                                    
                                    # Calculate total distance for the optimized route
                                    if use_real_roads and route_paths:
                                        # Calculate distance using the real road paths
                                        total_distance = 0
                                        for path in route_paths:
                                            for i in range(len(path) - 1):
                                                segment_distance = optimizer._haversine_distance(
                                                    path[i][0], path[i][1],  # lat1, lon1
                                                    path[i+1][0], path[i+1][1]  # lat2, lon2
                                                )
                                                total_distance += segment_distance
                                    else:
                                        # Calculate direct distances
                                        total_distance = 0
                                        for i in range(len(optimized_route) - 1):
                                            distance = optimizer._haversine_distance(
                                                optimized_route[i]['latitude'], optimized_route[i]['longitude'],
                                                optimized_route[i+1]['latitude'], optimized_route[i+1]['longitude']
                                            )
                                            total_distance += distance
                                    
                                    # Convert to appropriate units
                                    if st.session_state.distance_units == "Miles":
                                        total_distance = total_distance / 1609.34  # meters to miles
                                        distance_unit = "miles"
                                    else:
                                        total_distance = total_distance / 1000  # meters to km
                                        distance_unit = "kilometers"
                                        
                                    st.info(f"Total route distance: {total_distance:.2f} {distance_unit}")
                                    st.rerun()
        else:
            st.info("No stops added yet. Add stops manually or import from a document.")
    
    # Map View tab
# Inside tab2 (Map View tab)
with tab2:
    st.header("Map View")
    
    # Map settings
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Center Map"):
            st.success("Map centered")
    
    with col2:
        if 'route_paths' in st.session_state and st.session_state.route_paths:
            road_view = st.checkbox("Show Road Paths", value=True, 
                                  help="Toggle between road paths and direct lines")
        else:
            road_view = False
    
    with col3:
        if st.button("Export Map as HTML"):
            if not st.session_state.stops:
                st.error("No stops available to export.")
            else:
                # Create map
                route = st.session_state.optimized_route if st.session_state.optimized_route else st.session_state.stops
                route_paths = st.session_state.route_paths if 'route_paths' in st.session_state and road_view else None
                
                if all('latitude' in stop and 'longitude' in stop for stop in route):
                    # Create map file
                    m = create_map(route, route_paths, st.session_state.map_provider)
                    
                    # Save map to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                        m.save(tmp.name)
                    
                    # Offer download
                    with open(tmp.name, 'rb') as f:
                        map_data = f.read()
                        st.download_button(
                            label="Download Map HTML",
                            data=map_data,
                            file_name="route_map.html",
                            mime="text/html"
                        )
                    
                    # Clean up
                    os.unlink(tmp.name)
                else:
                    st.error("Some stops are missing coordinates. Please geocode them first.")
    
    # Display map
    if st.session_state.stops:
        route = st.session_state.optimized_route if st.session_state.optimized_route else st.session_state.stops
        
        # Decide whether to show road paths or not
        route_paths = None
        if 'route_paths' in st.session_state and road_view:
            route_paths = st.session_state.route_paths
        
        # Check if we have coordinates for all stops
        if all('latitude' in stop and 'longitude' in stop for stop in route):
            st.subheader("Route Map")
            m = create_map(route, route_paths, st.session_state.map_provider)
            folium_static(m, width=1000, height=600)
            
            # Display basic route statistics
            if len(route) > 1:
                st.subheader("Route Statistics")
                
                optimizer = RouteOptimizer()
                total_distance = 0
                
                if route_paths:
                    # Calculate distance along the actual road paths
                    for path in route_paths:
                        for i in range(len(path) - 1):
                            segment_distance = optimizer._haversine_distance(
                                path[i][0], path[i][1],  # lat1, lon1
                                path[i+1][0], path[i+1][1]  # lat2, lon2
                            )
                            total_distance += segment_distance
                else:
                    # Calculate direct distances
                    for i in range(len(route) - 1):
                        total_distance += optimizer._haversine_distance(
                            route[i]['latitude'], route[i]['longitude'],
                            route[i+1]['latitude'], route[i+1]['longitude']
                        )
                
                # Convert to appropriate units
                if st.session_state.distance_units == "Miles":
                    total_distance = total_distance / 1609.34  # meters to miles
                    distance_unit = "miles"
                else:
                    total_distance = total_distance / 1000  # meters to km
                    distance_unit = "km"
                    
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stops", len(route))
                with col2:
                    st.metric(f"Total Distance ({distance_unit})", f"{total_distance:.2f}")
                with col3:
                    if route_paths:
                        st.metric("Routing Type", "Real Roads")
                    else:
                        st.metric("Routing Type", "Direct Lines")
                
                # Show driving directions if we have road paths
                if route_paths:
                    with st.expander("Driving Directions"):
                        st.write("### Turn-by-Turn Directions")
                        st.info("Actual turn-by-turn directions would require additional API calls to a directions service.")
                        
                        for i, stop in enumerate(route[:-1]):
                            st.write(f"**{i+1}. From {stop['name']} to {route[i+1]['name']}**")
                            st.write(f"   - Drive approximately {optimizer._haversine_distance(stop['latitude'], stop['longitude'], route[i+1]['latitude'], route[i+1]['longitude']) / (1609.34 if st.session_state.distance_units == 'Miles' else 1000):.2f} {distance_unit}")
        else:
            st.warning("Some stops are missing coordinates. Please geocode them first.")
    else:
        st.info("No stops added yet. Add stops manually or import from a document.")
    
    # Reports tab
with tab3:
        st.header("Route Reports")
        
        # Only show report options if we have stops
        if not st.session_state.stops:
            st.info("Add stops first to generate reports.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Report Type",
                    ["Standard Route Report", "Detailed Distance Matrix", "Summary Statistics"]
                )
            
            with col2:
                report_format = st.selectbox(
                    "Export Format",
                    ["Markdown", "CSV", "PDF"]
                )
            
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    route = st.session_state.optimized_route if st.session_state.optimized_route else st.session_state.stops
                    
                    # Check if we have coordinates for distance calculations
                    if report_type in ["Standard Route Report", "Detailed Distance Matrix"] and not all('latitude' in stop and 'longitude' in stop for stop in route):
                        st.warning("Some stops are missing coordinates. Distance calculations will be omitted.")
                    
                    # Generate the appropriate report
                    if report_type == "Standard Route Report":
                        report_content = generate_report()
                        
                        # Display the report directly in the UI
                        st.subheader("Route Report")
                        st.markdown(report_content)
                        
                        # Prepare for download
                        if report_format == "Markdown":
                            download_data = report_content.encode()
                            download_filename = f"route_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            download_mimetype = "text/markdown"
                        elif report_format == "CSV":
                            # Convert to CSV format
                            csv_rows = []
                            csv_rows.append(["Stop #", "Name", "Address", "Latitude", "Longitude", "Distance to Next"])
                            
                            optimizer = RouteOptimizer()
                            for i, stop in enumerate(route):
                                row = [
                                    i+1,
                                    stop.get('name', 'Unnamed'),
                                    stop.get('address', 'No address'),
                                    stop.get('latitude', ''),
                                    stop.get('longitude', '')
                                ]
                                
                                # Add distance to next stop
                                if i < len(route) - 1 and 'latitude' in stop and 'longitude' in stop and 'latitude' in route[i+1] and 'longitude' in route[i+1]:
                                    distance = optimizer._haversine_distance(
                                        stop['latitude'], stop['longitude'],
                                        route[i+1]['latitude'], route[i+1]['longitude']
                                    )
                                    
                                    # Convert to appropriate units
                                    if st.session_state.distance_units == "Miles":
                                        distance = distance / 1609.34  # meters to miles
                                    else:
                                        distance = distance / 1000  # meters to km
                                    
                                    row.append(f"{distance:.2f}")
                                else:
                                    row.append('')
                                
                                csv_rows.append(row)
                            
                            # Convert to CSV string
                            csv_buffer = io.StringIO()
                            csv_writer = csv.writer(csv_buffer)
                            csv_writer.writerows(csv_rows)
                            download_data = csv_buffer.getvalue().encode()
                            download_filename = f"route_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            download_mimetype = "text/csv"
                        elif report_format == "PDF":
                            # Create a temporary markdown file
                            with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
                                tmp.write(report_content.encode())
                                tmp_path = tmp.name
                            
                            # We would ideally use a library like weasyprint to convert markdown to PDF
                            # but for simplicity, just provide the markdown
                            download_data = report_content.encode()
                            download_filename = f"route_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            download_mimetype = "text/markdown"
                            
                            st.warning("PDF export is not fully implemented. Providing markdown format instead.")
                    
                    elif report_type == "Detailed Distance Matrix":
                        # Create a distance matrix between all stops
                        st.subheader("Distance Matrix")
                        
                        if all('latitude' in stop and 'longitude' in stop for stop in route):
                            optimizer = RouteOptimizer()
                            distance_matrix = optimizer._calculate_distance_matrix(route)
                            
                            # Convert to appropriate units
                            if st.session_state.distance_units == "Miles":
                                distance_matrix = distance_matrix / 1609.34  # meters to miles
                                distance_unit = "miles"
                            else:
                                distance_matrix = distance_matrix / 1000  # meters to km
                                distance_unit = "km"
                            
                            # Create a DataFrame with stop names
                            stop_names = [stop.get('name', f"Stop {i+1}") for i, stop in enumerate(route)]
                            df_distance = pd.DataFrame(distance_matrix, columns=stop_names, index=stop_names)
                            
                            # Display the matrix
                            st.dataframe(df_distance.style.format("{:.2f}"))
                            
                            # Prepare for download
                            if report_format == "CSV":
                                csv_buffer = io.StringIO()
                                df_distance.to_csv(csv_buffer)
                                download_data = csv_buffer.getvalue().encode()
                                download_filename = f"distance_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                download_mimetype = "text/csv"
                            else:
                                # For other formats, still provide CSV
                                csv_buffer = io.StringIO()
                                df_distance.to_csv(csv_buffer)
                                download_data = csv_buffer.getvalue().encode()
                                download_filename = f"distance_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                download_mimetype = "text/csv"
                                st.warning(f"{report_format} export is not supported for distance matrix. Providing CSV instead.")
                        else:
                            st.error("Cannot create distance matrix: some stops are missing coordinates.")
                            download_data = "Error: Missing coordinates".encode()
                            download_filename = "error.txt"
                            download_mimetype = "text/plain"
                    
                    elif report_type == "Summary Statistics":
                        # Generate summary statistics
                        st.subheader("Route Summary Statistics")
                        
                        stats = {
                            "Total Stops": len(route),
                            "Optimized Route": "Yes" if st.session_state.optimized_route else "No",
                            "Generated Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Calculate distance if we have coordinates
                        if all('latitude' in stop and 'longitude' in stop for stop in route):
                            optimizer = RouteOptimizer()
                            total_distance = 0
                            
                            for i in range(len(route) - 1):
                                total_distance += optimizer._haversine_distance(
                                    route[i]['latitude'], route[i]['longitude'],
                                    route[i+1]['latitude'], route[i+1]['longitude']
                                )
                            
                            # Convert to appropriate units
                            if st.session_state.distance_units == "Miles":
                                total_distance = total_distance / 1609.34  # meters to miles
                                distance_unit = "miles"
                            else:
                                total_distance = total_distance / 1000  # meters to km
                                distance_unit = "km"
                                
                            stats["Total Distance"] = f"{total_distance:.2f} {distance_unit}"
                            
                            # Estimate average distance between stops
                            avg_distance = total_distance / (len(route) - 1) if len(route) > 1 else 0
                            stats["Average Distance Between Stops"] = f"{avg_distance:.2f} {distance_unit}"
                        
                        # Display statistics
                        for key, value in stats.items():
                            st.text(f"{key}: {value}")
                        
                        # Prepare for download
                        if report_format == "CSV":
                            csv_buffer = io.StringIO()
                            csv_writer = csv.writer(csv_buffer)
                            csv_writer.writerow(["Metric", "Value"])
                            for key, value in stats.items():
                                csv_writer.writerow([key, value])
                            download_data = csv_buffer.getvalue().encode()
                            download_filename = f"route_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            download_mimetype = "text/csv"
                        elif report_format == "Markdown":
                            md_content = "# Route Summary Statistics\n\n"
                            for key, value in stats.items():
                                md_content += f"**{key}**: {value}\n\n"
                            download_data = md_content.encode()
                            download_filename = f"route_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            download_mimetype = "text/markdown"
                        else:
                            # Default to markdown for unsupported formats
                            md_content = "# Route Summary Statistics\n\n"
                            for key, value in stats.items():
                                md_content += f"**{key}**: {value}\n\n"
                            download_data = md_content.encode()
                            download_filename = f"route_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            download_mimetype = "text/markdown"
                            st.warning(f"{report_format} export is not fully implemented. Providing markdown format instead.")
                    
                    # Offer download option
                    st.download_button(
                        label=f"Download Report as {report_format}",
                        data=download_data,
                        file_name=download_filename,
                        mime=download_mimetype
                    )
            
            # Display a preview of the route
            if st.session_state.stops:
                st.subheader("Route Preview")
                
                route = st.session_state.optimized_route if st.session_state.optimized_route else st.session_state.stops
                
                # Create a simple table of stops
                preview_data = []
                for i, stop in enumerate(route):
                    preview_data.append({
                        "Stop #": i+1,
                        "Name": stop.get('name', 'Unnamed'),
                        "Address": stop.get('address', 'No address')
                    })
                
                st.table(pd.DataFrame(preview_data))
                
                # Add a mini-map if we have coordinates
                if all('latitude' in stop and 'longitude' in stop for stop in route):
                    st.subheader("Route Mini-Map")
                    mini_map = create_map(route, st.session_state.map_provider)
                    folium_static(mini_map, width=800, height=400)

# Run the app - this isn't needed when deploying with streamlit
if __name__ == "__main__":
    pass
