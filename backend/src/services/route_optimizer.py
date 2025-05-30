import numpy as np
from typing import List, Dict, Any, Tuple
import math
import logging
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Service for finding the optimal route through a set of locations.
    Implements various TSP (Traveling Salesman Problem) algorithms.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the route optimizer.
        
        Args:
            api_keys: Dictionary with provider names as keys and API keys as values
        """
        self.api_keys = api_keys or {}
        
        # Load API keys from environment variables if not provided
        if not self.api_keys.get('google'):
            self.api_keys['google'] = os.getenv('GOOGLE_MAPS_API_KEY', '')
    
    def optimize_route(self, locations: List[Dict[str, Any]], 
                       start_index: int = 0,
                       algorithm: str = 'nearest_neighbor') -> List[Dict[str, Any]]:
        """
        Find the optimal route through all locations.
        
        Args:
            locations: List of location dictionaries with 'latitude' and 'longitude'
            start_index: Index of the starting location
            algorithm: Algorithm to use ('nearest_neighbor', 'genetic', or 'exact')
            
        Returns:
            Reordered list of locations for the optimal route
        """
        if not locations:
            return []
        
        # Ensure we have coordinates for all locations
        if not all('latitude' in loc and 'longitude' in loc for loc in locations):
            logger.error("Not all locations have coordinates")
            return locations
        
        # Calculate the distance matrix
        distance_matrix = self._calculate_distance_matrix(locations)
        
        # Apply the chosen algorithm
        if algorithm == 'nearest_neighbor':
            route_indices = self._nearest_neighbor_tsp(distance_matrix, start_index)
        elif algorithm == 'genetic':
            route_indices = self._genetic_algorithm_tsp(distance_matrix, start_index)
        elif algorithm == 'exact':
            route_indices = self._exact_tsp(distance_matrix, start_index)
        else:
            route_indices = self._nearest_neighbor_tsp(distance_matrix, start_index)
        
        # Reorder locations based on the route
        optimized_route = [locations[i] for i in route_indices]
        return optimized_route
    
    def optimize_route_with_google(self, locations: List[Dict[str, Any]], 
                                  start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Find the optimal route using Google's Directions API.
        Note: Google API has limits on the number of waypoints (23 for standard API).
        
        Args:
            locations: List of location dictionaries with 'latitude' and 'longitude'
            start_index: Index of the starting location
            
        Returns:
            Reordered list of locations for the optimal route
        """
        if not self.api_keys.get('google'):
            logger.warning("No Google API key provided, falling back to local optimization")
            return self.optimize_route(locations, start_index)
        
        if len(locations) > 25:  # Google API limit is 23 waypoints plus origin and destination
            logger.warning("Too many locations for Google API, falling back to local optimization")
            return self.optimize_route(locations, start_index)
        
        try:
            # Prepare waypoints
            waypoints = []
            for i, loc in enumerate(locations):
                if i != start_index:  # Skip the origin
                    waypoints.append(f"{loc['latitude']},{loc['longitude']}")
            
            origin = f"{locations[start_index]['latitude']},{locations[start_index]['longitude']}"
            destination = origin  # Return to the starting point
            
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                'origin': origin,
                'destination': destination,
                'waypoints': f"optimize:true|{('|').join(waypoints)}",
                'key': self.api_keys['google']
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                # Extract the optimized waypoint order
                order = data['routes'][0]['waypoint_order']
                
                # Convert to route with starting point first
                route_indices = [start_index]
                for idx in order:
                    # Adjust index to account for skipping the origin
                    adjusted_idx = idx if idx < start_index else idx + 1
                    route_indices.append(adjusted_idx)
                
                # Reorder locations based on the route
                optimized_route = [locations[i] for i in route_indices]
                return optimized_route
            else:
                logger.warning(f"Google Directions API error: {data['status']}")
                return self.optimize_route(locations, start_index)
                
        except Exception as e:
            logger.warning(f"Error using Google Directions API: {e}")
            return self.optimize_route(locations, start_index)
    
    def _calculate_distance_matrix(self, locations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate the distance matrix between all locations.
        
        Args:
            locations: List of location dictionaries with 'latitude' and 'longitude'
            
        Returns:
            2D numpy array with distances between all pairs of locations
        """
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        # Try to use Google Distance Matrix API for more accurate distances if API key available
        if self.api_keys.get('google') and n <= 25:  # Google has limits
            try:
                return self._get_google_distance_matrix(locations)
            except Exception as e:
                logger.warning(f"Error using Google Distance Matrix API: {e}")
                logger.warning("Falling back to haversine distances")
        
        # Fall back to haversine distance (as the crow flies)
        for i in range(n):
            for j in range(i+1, n):
                distance = self._haversine_distance(
                    locations[i]['latitude'], locations[i]['longitude'],
                    locations[j]['latitude'], locations[j]['longitude']
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _get_google_distance_matrix(self, locations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Get distance matrix using Google Distance Matrix API.
        
        Args:
            locations: List of location dictionaries with 'latitude' and 'longitude'
            
        Returns:
            2D numpy array with distances between all pairs of locations
        """
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        # Convert locations to formatted strings
        formatted_locations = [
            f"{loc['latitude']},{loc['longitude']}" for loc in locations
        ]
        
        # Google Distance Matrix API has a limit of 25 origins and 25 destinations per request
        # We'll process the matrix in chunks if needed
        max_chunk_size = 10  # Stay well below limits
        
        for i in range(0, n, max_chunk_size):
            origins_chunk = formatted_locations[i:i+max_chunk_size]
            
            for j in range(0, n, max_chunk_size):
                dests_chunk = formatted_locations[j:j+max_chunk_size]
                
                url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    'origins': '|'.join(origins_chunk),
                    'destinations': '|'.join(dests_chunk),
                    'key': self.api_keys['google'],
                    'mode': 'driving'
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if data['status'] == 'OK':
                    for idx_i, row in enumerate(data['rows']):
                        for idx_j, element in enumerate(row['elements']):
                            if element['status'] == 'OK':
                                # Use distance in meters
                                distance_matrix[i + idx_i, j + idx_j] = element['distance']['value']
                            else:
                                # Fall back to haversine if Google can't calculate
                                loc_i = locations[i + idx_i]
                                loc_j = locations[j + idx_j]
                                distance_matrix[i + idx_i, j + idx_j] = self._haversine_distance(
                                    loc_i['latitude'], loc_i['longitude'],
                                    loc_j['latitude'], loc_j['longitude']
                                )
                else:
                    raise ValueError(f"Google Distance Matrix API error: {data['status']}")
                
                # Sleep to avoid hitting API rate limits
                time.sleep(0.2)
        
        return distance_matrix
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on the Earth.
        
        Args:
            lat1, lon1: Coordinates of first point in degrees
            lat2, lon2: Coordinates of second point in degrees
            
        Returns:
            Distance in meters
        """
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
    
    def _nearest_neighbor_tsp(self, distance_matrix: np.ndarray, start_index: int) -> List[int]:
        """
        Solve the TSP using the Nearest Neighbor heuristic.
        
        Args:
            distance_matrix: 2D array of distances between all locations
            start_index: Index of the starting location
            
        Returns:
            List of indices representing the order to visit locations
        """
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
    
    def _exact_tsp(self, distance_matrix: np.ndarray, start_index: int) -> List[int]:
        """
        Solve the TSP exactly using dynamic programming.
        Only practical for small numbers of locations (up to ~15).
        
        Args:
            distance_matrix: 2D array of distances between all locations
            start_index: Index of the starting location
            
        Returns:
            List of indices representing the order to visit locations
        """
        n = distance_matrix.shape[0]
        
        # If too many locations, fall back to nearest neighbor
        if n > 15:
            logger.warning("Too many locations for exact TSP solution, using nearest neighbor")
            return self._nearest_neighbor_tsp(distance_matrix, start_index)
        
        # Initialize memoization table
        # dp[i][mask] = (distance, path) where:
        # - i is the current city
        # - mask is a bitmask of visited cities
        # - distance is the shortest distance found
        # - path is the path taken
        dp = {}
        
        # Helper function for recursive DP with memoization
        def tsp_dp(i, mask):
            if mask == ((1 << n) - 1):
                # All cities visited, return to start
                return distance_matrix[i, start_index], [i, start_index]
            
            if (i, mask) in dp:
                return dp[(i, mask)]
            
            min_distance = float('inf')
            best_path = []
            
            for j in range(n):
                if not (mask & (1 << j)):  # If j has not been visited
                    # Visit j
                    new_mask = mask | (1 << j)
                    distance, path = tsp_dp(j, new_mask)
                    total_distance = distance_matrix[i, j] + distance
                    
                    if total_distance < min_distance:
                        min_distance = total_distance
                        best_path = [i] + path
            
            dp[(i, mask)] = (min_distance, best_path)
            return min_distance, best_path
        
        # Start from the given start index
        _, path = tsp_dp(start_index, 1 << start_index)
        
        # Remove the duplicate of the starting point at the end
        path.pop()
        
        return path
    
    def _genetic_algorithm_tsp(self, distance_matrix: np.ndarray, start_index: int,
                              population_size: int = 100, generations: int = 100) -> List[int]:
        """
        Solve the TSP using a genetic algorithm.
        Good for medium to large numbers of locations.
        
        Args:
            distance_matrix: 2D array of distances between all locations
            start_index: Index of the starting location
            population_size: Size of the population in the genetic algorithm
            generations: Number of generations to evolve
            
        Returns:
            List of indices representing the order to visit locations
        """
        n = distance_matrix.shape[0]
        
        # Initialize population
        population = []
        for _ in range(population_size):
            # Create a random route starting with start_index
            route = list(range(n))
            route.remove(start_index)
            np.random.shuffle(route)
            route.insert(0, start_index)
            population.append(route)
        
        def fitness(route):
            """Calculate the fitness (negative total distance) of a route."""
            total_distance = 0
            for i in range(n - 1):
                total_distance += distance_matrix[route[i], route[i+1]]
            # Add the distance back to the start
            total_distance += distance_matrix[route[-1], route[0]]
            return -total_distance
        
        def crossover(parent1, parent2):
            """Create a child route by combining segments from both parents."""
            # Select a random segment
            start, end = sorted(np.random.choice(range(1, n), 2, replace=False))
            
            # Inherit segment from parent1
            child = [-1] * n
            child[0] = start_index  # Always start with the start_index
            
            # Copy segment from parent1
            for i in range(start, end + 1):
                child[i] = parent1[i]
            
            # Fill in remaining positions from parent2 in order
            j = 1
            for i in range(1, n):
                if j == start:
                    j = end + 1
                if child[j] == -1:
                    # Get the next city from parent2 that isn't already in child
                    for city in parent2:
                        if city not in child:
                            child[j] = city
                            break
                    j += 1
            
            return child
        
        def mutate(route, mutation_rate=0.1):
            """Randomly swap two cities in the route with probability mutation_rate."""
            if np.random.random() < mutation_rate:
                # Don't swap the start index at position 0
                i, j = np.random.choice(range(1, n), 2, replace=False)
                route[i], route[j] = route[j], route[i]
            return route
        
        # Evolve the population
        for _ in range(generations):
            # Calculate fitness for each route
            fitness_scores = [fitness(route) for route in population]
            
            # Select parents based on fitness (higher fitness = more likely)
            parents_indices = np.random.choice(
                range(population_size),
                size=population_size,
                p=np.exp(fitness_scores) / np.sum(np.exp(fitness_scores))
            )
            parents = [population[i] for i in parents_indices]
            
            # Create a new population through crossover and mutation
            new_population = [population[np.argmax(fitness_scores)]]  # Keep the best route
            
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Return the best route found
        fitness_scores = [fitness(route) for route in population]
        best_route = population[np.argmax(fitness_scores)]
        
        return best_route