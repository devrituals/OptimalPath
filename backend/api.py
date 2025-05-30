from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import datetime
import math
import requests

# Import or copy the RouteOptimizer class from app.py
# For now, I'll copy the minimal RouteOptimizer logic needed for optimization

class Stop(BaseModel):
    name: str
    address: Optional[str] = None
    latitude: float
    longitude: float
    is_current_location: Optional[bool] = False

class OptimizeRouteRequest(BaseModel):
    stops: List[Stop]
    current_location: Optional[Stop] = None
    use_real_roads: Optional[bool] = False
    cost_type: Optional[str] = "distance"
    api_key: Optional[str] = None

class OptimizeRouteResponse(BaseModel):
    optimized_route: List[Stop]
    message: Optional[str] = None

class GeocodeRequest(BaseModel):
    address: str

class GeocodeResponse(BaseModel):
    latitude: float
    longitude: float
    display_name: str

class DirectionsRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

class DirectionsStep(BaseModel):
    instruction: str
    distance: float
    duration: float

class DirectionsResponse(BaseModel):
    steps: list
    total_distance: float
    total_duration: float

app = FastAPI()

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal RouteOptimizer logic (copy from app.py)
class RouteOptimizer:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _calculate_distance_matrix(self, locations):
        n = len(locations)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._haversine_distance(
                        locations[i].latitude, locations[i].longitude,
                        locations[j].latitude, locations[j].longitude
                    )
        return matrix

    def _nearest_neighbor_tsp(self, distance_matrix, start_index):
        n = len(distance_matrix)
        visited = [False] * n
        route = [start_index]
        visited[start_index] = True
        for _ in range(n - 1):
            last = route[-1]
            next_city = min(
                [(i, distance_matrix[last][i]) for i in range(n) if not visited[i]],
                key=lambda x: x[1],
            )[0]
            route.append(next_city)
            visited[next_city] = True
        return route

    def optimize_route(self, locations, start_index=0):
        distance_matrix = self._calculate_distance_matrix(locations)
        order = self._nearest_neighbor_tsp(distance_matrix, start_index)
        return [locations[i] for i in order]

@app.post("/optimize_route", response_model=OptimizeRouteResponse)
def optimize_route(request: OptimizeRouteRequest):
    stops = request.stops
    current_location = request.current_location
    if current_location:
        locations = [current_location] + stops
        start_index = 0
    else:
        locations = stops
        start_index = 0
    optimizer = RouteOptimizer(api_key=request.api_key)
    optimized = optimizer.optimize_route(locations, start_index=start_index)
    return OptimizeRouteResponse(optimized_route=optimized, message="Route optimized successfully.")

@app.post("/geocode", response_model=GeocodeResponse)
def geocode(request: GeocodeRequest):
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        'q': request.address,
        'format': 'json',
        'limit': 1
    }
    response = requests.get(url, params=params, headers={"User-Agent": "OptimalPathApp/1.0"})
    if response.status_code != 200 or not response.json():
        raise HTTPException(status_code=404, detail="Address not found")
    data = response.json()[0]
    return GeocodeResponse(latitude=float(data['lat']), longitude=float(data['lon']), display_name=data['display_name'])

@app.post("/directions", response_model=DirectionsResponse)
def directions(request: DirectionsRequest):
    # Dummy implementation: just return a single step
    step = DirectionsStep(
        instruction=f"Go from ({request.start_lat}, {request.start_lon}) to ({request.end_lat}, {request.end_lon})",
        distance=RouteOptimizer()._haversine_distance(request.start_lat, request.start_lon, request.end_lat, request.end_lon),
        duration=0.0
    )
    return DirectionsResponse(steps=[step], total_distance=step.distance, total_duration=step.duration)

# --- Models ---
class GeocodingRequest(BaseModel):
    address: str

class GeocodingBatchRequest(BaseModel):
    addresses: List[str]

class GeocodingResponse(BaseModel):
    latitude: float
    longitude: float
    display_name: str

class Location(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    latitude: float
    longitude: float

class RouteRequest(BaseModel):
    locations: List[Location]
    current_location: Optional[Location] = None

class RouteResponse(BaseModel):
    optimized_route: List[Location]
    message: Optional[str] = None

class TurnByTurnDirections(BaseModel):
    steps: List[str]
    total_distance: float
    total_duration: float

class UploadLocationsRequest(BaseModel):
    locations: List[Location]

class LocationTrackingRequest(BaseModel):
    user_id: str
    location: Location

class LocationTrackingResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- Endpoints ---
@app.post("/geocode", response_model=GeocodingResponse)
def geocode_address(request: GeocodingRequest):
    """Geocode a single address."""
    # Placeholder implementation
    return GeocodingResponse(latitude=0.0, longitude=0.0, display_name=request.address)

@app.post("/geocode/batch", response_model=List[GeocodingResponse])
def batch_geocode(request: GeocodingBatchRequest):
    """Geocode multiple addresses."""
    # Placeholder implementation
    return [GeocodingResponse(latitude=0.0, longitude=0.0, display_name=addr) for addr in request.addresses]

@app.post("/optimize-route", response_model=RouteResponse)
def optimize_route(request: RouteRequest):
    """Optimize a route given a list of locations."""
    # Placeholder implementation
    return RouteResponse(optimized_route=request.locations, message="Route optimized (placeholder)")

@app.post("/upload-locations")
def upload_locations(request: UploadLocationsRequest):
    """Upload a list of locations."""
    # Placeholder implementation
    return {"status": "success", "message": "Locations uploaded (placeholder)"}

@app.post("/tracking/start", response_model=LocationTrackingResponse)
def start_tracking(request: LocationTrackingRequest):
    """Start tracking a user/device."""
    # Placeholder implementation
    return LocationTrackingResponse(status="started", message="Tracking started (placeholder)")

@app.post("/tracking/update", response_model=LocationTrackingResponse)
def update_tracking(request: LocationTrackingRequest):
    """Update the tracked location."""
    # Placeholder implementation
    return LocationTrackingResponse(status="updated", message="Location updated (placeholder)")

@app.get("/tracking/status", response_model=LocationTrackingResponse)
def get_tracking_status(user_id: str):
    """Get current tracking status."""
    # Placeholder implementation
    return LocationTrackingResponse(status="active", message="Tracking status (placeholder)")

@app.post("/tracking/stop", response_model=LocationTrackingResponse)
def stop_tracking(request: LocationTrackingRequest):
    """Stop tracking a user/device."""
    # Placeholder implementation
    return LocationTrackingResponse(status="stopped", message="Tracking stopped (placeholder)")

@app.post("/directions", response_model=TurnByTurnDirections)
def get_directions(request: DirectionsRequest):
    """Get turn-by-turn directions between two points."""
    # Placeholder implementation
    return TurnByTurnDirections(steps=["Go straight", "Turn left"], total_distance=1.0, total_duration=2.0)

@app.get("/map")
def get_map():
    """Get a map (static or dynamic)."""
    # Placeholder implementation
    return {"map_url": "https://via.placeholder.com/600x400.png?text=Map+Placeholder"} 