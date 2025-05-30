import requests
from typing import Dict, Any, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)

class GeocodingService:
    """
    Service for geocoding addresses to obtain coordinates.
    Uses multiple geocoding providers with fallbacks.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the geocoding service with API keys.
        
        Args:
            api_keys: Dictionary with provider names as keys and API keys as values
        """
        self.api_keys = api_keys or {}
        
        # Load API keys from environment variables if not provided
        if not self.api_keys.get('google'):
            self.api_keys['google'] = os.getenv('GOOGLE_MAPS_API_KEY', '')
        if not self.api_keys.get('mapbox'):
            self.api_keys['mapbox'] = os.getenv('MAPBOX_API_KEY', '')
        if not self.api_keys.get('opencage'):
            self.api_keys['opencage'] = os.getenv('OPENCAGE_API_KEY', '')
    
    def geocode_location(self, location: Dict[str, Any]) -> Dict[str, Any]:
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
        if 'address' not in location:
            return location
        
        address = location['address']
        if 'name' in location:
            query = f"{location['name']}, {address}"
        else:
            query = address
        
        # Try different geocoding services
        coordinates = (
            self._geocode_with_google(query) or
            self._geocode_with_mapbox(query) or
            self._geocode_with_opencage(query) or
            self._geocode_with_nominatim(query)
        )
        
        if coordinates:
            location['latitude'], location['longitude'] = coordinates
        
        return location
    
    def _geocode_with_google(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode using Google Maps API."""
        if not self.api_keys.get('google'):
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': address,
                'key': self.api_keys['google']
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return location['lat'], location['lng']
        except Exception as e:
            logger.warning(f"Google geocoding error: {e}")
        
        return None
    
    def _geocode_with_mapbox(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode using Mapbox API."""
        if not self.api_keys.get('mapbox'):
            return None
        
        try:
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
            params = {
                'access_token': self.api_keys['mapbox'],
                'limit': 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('features') and len(data['features']) > 0:
                coordinates = data['features'][0]['center']
                # Mapbox returns [lng, lat]
                return coordinates[1], coordinates[0]
        except Exception as e:
            logger.warning(f"Mapbox geocoding error: {e}")
        
        return None
    
    def _geocode_with_opencage(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode using OpenCage API."""
        if not self.api_keys.get('opencage'):
            return None
        
        try:
            url = "https://api.opencagedata.com/geocode/v1/json"
            params = {
                'q': address,
                'key': self.api_keys['opencage'],
                'limit': 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                geometry = data['results'][0]['geometry']
                return geometry['lat'], geometry['lng']
        except Exception as e:
            logger.warning(f"OpenCage geocoding error: {e}")
        
        return None
    
    def _geocode_with_nominatim(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode using Nominatim (OpenStreetMap) API - no API key required."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1
            }
            headers = {
                'User-Agent': 'LocationOptimizer/1.0'  # Required by Nominatim
            }
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if data and len(data) > 0:
                return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            logger.warning(f"Nominatim geocoding error: {e}")
        
        return None