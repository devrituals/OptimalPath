import { useState } from 'react';
import { optimizeRoute, geocodeAddress, getDirections } from '../services/api';

export const useRouteOptimization = () => {
  const [locations, setLocations] = useState([]);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [optimizedRoute, setOptimizedRoute] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const addLocation = async (address) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      const geocodedLocation = await geocodeAddress(address);
      setLocations(prev => [...prev, {
        name: address,
        address,
        latitude: geocodedLocation.latitude,
        longitude: geocodedLocation.longitude,
      }]);
      setSuccess('Location added successfully!');
    } catch (err) {
      setError('Failed to add location: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const removeLocation = (index) => {
    setLocations(prev => prev.filter((_, i) => i !== index));
  };

  const setCurrentLocationFromAddress = async (address) => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      const geocodedLocation = await geocodeAddress(address);
      setCurrentLocation({
        name: 'Current Location',
        address,
        latitude: geocodedLocation.latitude,
        longitude: geocodedLocation.longitude,
      });
      setSuccess('Current location set!');
    } catch (err) {
      setError('Failed to set current location: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const optimizeCurrentRoute = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      const result = await optimizeRoute(locations, currentLocation);
      setOptimizedRoute(result.optimized_route);
      setSuccess('Route optimized successfully!');
      return result;
    } catch (err) {
      setError('Failed to optimize route: ' + err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const getRouteDirections = async () => {
    if (optimizedRoute.length < 2) return [];
    const directions = [];
    for (let i = 0; i < optimizedRoute.length - 1; i++) {
      const current = optimizedRoute[i];
      const next = optimizedRoute[i + 1];
      const direction = await getDirections(current, next);
      directions.push(direction);
    }
    return directions;
  };

  return {
    locations,
    currentLocation,
    optimizedRoute,
    loading,
    error,
    success,
    addLocation,
    removeLocation,
    setCurrentLocationFromAddress,
    optimizeCurrentRoute,
    getRouteDirections,
    setLocations,
    setOptimizedRoute,
  };
}; 