from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from road_network import RoadNetwork
from optimal_path import OptimalPathFinder
import json

app = FastAPI(title="Optimal Path API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Location(BaseModel):
    latitude: float
    longitude: float
    name: Optional[str] = None

class PathRequest(BaseModel):
    locations: List[Location]
    start_time: Optional[str] = None
    end_time: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Optimal Path API is running"}

@app.post("/calculate-path")
async def calculate_path(request: PathRequest):
    try:
        # Initialize road network
        road_network = RoadNetwork()
        
        # Convert locations to the format expected by the path finder
        locations = [(loc.latitude, loc.longitude) for loc in request.locations]
        
        # Initialize path finder
        path_finder = OptimalPathFinder(road_network)
        
        # Calculate optimal path
        optimal_path = path_finder.find_optimal_path(locations)
        
        # Convert the path to a format suitable for the frontend
        path_coordinates = []
        for point in optimal_path:
            path_coordinates.append({
                "latitude": point[0],
                "longitude": point[1]
            })
        
        return {
            "status": "success",
            "path": path_coordinates,
            "total_distance": path_finder.get_total_distance(),
            "estimated_time": path_finder.get_estimated_time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 