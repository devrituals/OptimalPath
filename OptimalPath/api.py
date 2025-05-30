from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# Import existing classes and functions
from app import (
    LocationTrackingService,
    OptimalPathHandler,
    LocationParser,
    GeocodingService,
    RouteOptimizer
)

# Load environment variables
load_dotenv()

app = FastAPI(title="Route Optimizer API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your mobile app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Location(BaseModel):
    name: str
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class RouteRequest(BaseModel):
    locations: List[Location]
    use_real_roads: bool = True
    cost_type: str = "distance"
    start_at_current_location: bool = False

class RouteResponse(BaseModel):
    optimized_route: List[Dict[str, Any]]
    route_paths: List[Dict[str, Any]]
    total_distance: float
    estimated_time: float

class UserLocation(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime

# Initialize services
location_tracking = LocationTrackingService()
path_handler = OptimalPathHandler()
geocoding_service = GeocodingService()
route_optimizer = RouteOptimizer()

# API Routes
@app.post("/api/optimize-route")
async def optimize_route(request: RouteRequest):
    try:
        # Convert locations to the format expected by the optimizer
        locations = [
            {
                "name": loc.name,
                "address": loc.address,
                "latitude": loc.latitude,
                "longitude": loc.longitude
            }
            for loc in request.locations
        ]

        # Optimize route
        optimized_route, route_paths = route_optimizer.optimize_route(
            locations=locations,
            use_real_roads=request.use_real_roads,
            cost_type=request.cost_type,
            start_at_current_location=request.start_at_current_location
        )

        # Calculate total distance and time
        total_distance = 0
        total_time = 0
        for path in route_paths:
            total_distance += path.get("distance", 0)
            total_time += path.get("duration", 0)

        return RouteResponse(
            optimized_route=optimized_route,
            route_paths=route_paths,
            total_distance=total_distance,
            estimated_time=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-locations")
async def upload_locations(file: UploadFile = File(...)):
    try:
        parser = LocationParser()
        locations = await parser.parse_document(file)
        return {"locations": locations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-current-location")
async def update_current_location(location: UserLocation):
    try:
        # Update the current location in the path handler
        current_location = {
            "latitude": location.latitude,
            "longitude": location.longitude
        }
        
        if hasattr(st.session_state, 'optimized_route'):
            success = path_handler.update_optimal_path(
                st.session_state.optimized_route,
                current_location
            )
            if success:
                return {"status": "success", "message": "Route updated successfully"}
        
        return {"status": "success", "message": "Location updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/next-stop-info")
async def get_next_stop_info():
    try:
        distance, eta = path_handler.get_distance_to_next_stop()
        return {
            "distance": distance,
            "eta_minutes": eta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-tracking")
async def start_tracking():
    try:
        success = location_tracking.start_tracking()
        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-tracking")
async def stop_tracking():
    try:
        success = location_tracking.stop_tracking()
        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tracking-status")
async def get_tracking_status():
    try:
        status = location_tracking.get_tracking_status()
        return {"status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 