import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Update if needed

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const geocodeAddress = async (address) => {
  const response = await api.post('/geocode', { address });
  return response.data;
};

export const batchGeocode = async (addresses) => {
  const response = await api.post('/geocode/batch', { addresses });
  return response.data;
};

export const optimizeRoute = async (locations, currentLocation = null) => {
  const response = await api.post('/optimize-route', {
    locations,
    current_location: currentLocation,
  });
  return response.data;
};

export const getDirections = async (start, end) => {
  const response = await api.post('/directions', {
    start,
    end,
  });
  return response.data;
};

export const getMap = async () => {
  const response = await api.get('/map');
  return response.data;
}; 