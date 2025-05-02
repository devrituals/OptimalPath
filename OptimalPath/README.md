# Marrakech Interactive Route Planner

An advanced route planning application that helps users navigate through Marrakech efficiently. The application provides both graph-based and road-based routing capabilities, allowing users to find optimal paths between locations and plan multi-stop routes.

## Features

- **Interactive Map Interface**: Visualize routes on an interactive map powered by Folium
- **Dual Routing Modes**:
  - Graph-based routing for quick path finding
  - Road-based routing using actual street networks (powered by OSMnx)
- **Multiple Optimization Options**:
  - Optimize routes by distance or time
  - Support for multi-stop route planning
  - Traveling Salesman Problem (TSP) solver for optimal route ordering
- **Smart Caching**: Efficient caching of road networks for improved performance
- **Real-time Path Visualization**: Dynamic display of routes with distance and time information

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Folium**: Interactive map visualization
- **NetworkX**: Graph operations and path finding
- **OSMnx**: OpenStreetMap data integration
- **OR-Tools**: Route optimization and TSP solving

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

### Finding the Shortest Path

1. Select the "Shortest Path" tab
2. Choose your starting location and destination
3. Select optimization preference (distance or time)
4. Click "Find Shortest Path" to view the route

### Planning Multi-Stop Routes

1. Select the "Multi-Stop Route" tab
2. Choose multiple locations to visit
3. Select the delivery person's starting location
4. Choose routing type (Graph-based or Road-based)
5. Click "Find Optimal Route" to get the optimized route

## Features in Detail

### Road Network Integration

The application uses OSMnx to fetch and process real road network data from OpenStreetMap. This ensures that routes follow actual streets and paths in Marrakech.

### Caching System

To improve performance, the application implements a smart caching system for road networks. Downloaded map data is stored locally and reused in subsequent sessions.

### Route Optimization

The application offers two types of optimization:
- **Distance-based**: Find the shortest physical route
- **Time-based**: Find the quickest route considering road types and speed limits

### Interactive Visualization

- Real-time route display on an interactive map
- Clear markers for start and end points
- Color-coded path visualization
- Distance and time information for each route
