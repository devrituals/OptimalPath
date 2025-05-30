import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, ActivityIndicator, StyleSheet } from 'react-native';
import { useRouteOptimization } from '../hooks/useRouteOptimization';
import { Ionicons } from '@expo/vector-icons';
import MapView from './MapView';

const DEFAULT_CENTER = [31.634, -7.999]; // Marrakech

const RouteOptimizer = () => {
  const [newLocation, setNewLocation] = useState('');
  const [currentLocationInput, setCurrentLocationInput] = useState('');
  const {
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
    setLocations,
    setOptimizedRoute,
  } = useRouteOptimization();

  const handleAddLocation = async () => {
    if (newLocation.trim()) {
      await addLocation(newLocation.trim());
      setNewLocation('');
    }
  };

  const handleSetCurrentLocation = async () => {
    if (currentLocationInput.trim()) {
      await setCurrentLocationFromAddress(currentLocationInput.trim());
      setCurrentLocationInput('');
    }
  };

  // Map markers and route
  const allMarkers = [
    ...(currentLocation ? [currentLocation] : []),
    ...locations,
  ];
  const routeMarkers = optimizedRoute.length > 0 ? optimizedRoute : allMarkers;
  const polylinePositions = routeMarkers.map(loc => [loc.latitude, loc.longitude]);
  const mapCenter = polylinePositions.length > 0 ? polylinePositions[0] : DEFAULT_CENTER;

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Current Location Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Current Location</Text>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              placeholder="Enter your current location"
              value={currentLocationInput}
              onChangeText={setCurrentLocationInput}
            />
            <TouchableOpacity
              style={styles.addButton}
              onPress={handleSetCurrentLocation}
              disabled={loading}
            >
              <Ionicons name="location" size={24} color="white" />
            </TouchableOpacity>
          </View>
          {currentLocation && (
            <View style={styles.locationCard}>
              <Text style={styles.locationText}>{currentLocation.address}</Text>
            </View>
          )}
        </View>

        {/* Locations Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Add Locations</Text>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              placeholder="Enter location address"
              value={newLocation}
              onChangeText={setNewLocation}
            />
            <TouchableOpacity
              style={styles.addButton}
              onPress={handleAddLocation}
              disabled={loading}
            >
              <Ionicons name="add" size={24} color="white" />
            </TouchableOpacity>
          </View>

          {locations.map((loc, index) => (
            <View key={index} style={styles.stopCard}>
              <Text style={styles.stopText}>{loc.address}</Text>
              <TouchableOpacity
                style={styles.removeButton}
                onPress={() => removeLocation(index)}
                disabled={loading}
              >
                <Ionicons name="close-circle" size={24} color="#ff4444" />
              </TouchableOpacity>
            </View>
          ))}
        </View>

        {/* Optimized Route Section */}
        {optimizedRoute.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Optimized Route</Text>
            {optimizedRoute.map((loc, index) => (
              <View key={index} style={styles.routeCard}>
                <View style={styles.routeNumber}>
                  <Text style={styles.routeNumberText}>{index + 1}</Text>
                </View>
                <Text style={styles.routeText}>{loc.address}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Map Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Map</Text>
          <View style={{ height: 400, width: '100%', borderRadius: 8, overflow: 'hidden', borderWidth: 1, borderColor: '#ddd' }}>
            <MapView center={mapCenter} markers={routeMarkers} polyline={polylinePositions} />
          </View>
        </View>

        {/* Feedback Section */}
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        {success && (
          <View style={styles.successContainer}>
            <Text style={styles.successText}>{success}</Text>
          </View>
        )}
      </ScrollView>

      {/* Optimize Button */}
      <View style={styles.bottomContainer}>
        <TouchableOpacity
          style={[styles.optimizeButton, loading && styles.optimizeButtonDisabled]}
          onPress={optimizeCurrentRoute}
          disabled={loading || locations.length === 0}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.optimizeButtonText}>Optimize Route</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
    padding: 16,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  inputContainer: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  input: {
    flex: 1,
    height: 48,
    backgroundColor: 'white',
    borderRadius: 8,
    paddingHorizontal: 16,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  addButton: {
    width: 48,
    height: 48,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  locationCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  stopCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  routeCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
    marginBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  routeNumber: {
    width: 32,
    height: 32,
    backgroundColor: '#007AFF',
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  routeNumberText: {
    color: 'white',
    fontWeight: 'bold',
  },
  locationText: {
    fontSize: 16,
    color: '#333',
  },
  stopText: {
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  routeText: {
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  removeButton: {
    padding: 4,
  },
  bottomContainer: {
    padding: 16,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
  optimizeButton: {
    backgroundColor: '#007AFF',
    height: 48,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  optimizeButtonDisabled: {
    backgroundColor: '#ccc',
  },
  optimizeButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: {
    color: '#d32f2f',
    fontSize: 14,
  },
  successContainer: {
    backgroundColor: '#e8f5e9',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
  },
  successText: {
    color: '#388e3c',
    fontSize: 14,
  },
});

export default RouteOptimizer; 