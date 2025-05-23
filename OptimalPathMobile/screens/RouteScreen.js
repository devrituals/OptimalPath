import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';

export default function RouteScreen({ route }) {
  const { optimizedRoute } = route.params;

  // Calculate the initial region based on the first stop
  const initialRegion = {
    latitude: optimizedRoute[0].latitude,
    longitude: optimizedRoute[0].longitude,
    latitudeDelta: 0.0922,
    longitudeDelta: 0.0421,
  };

  // Create coordinates array for the polyline
  const coordinates = optimizedRoute.map(stop => ({
    latitude: stop.latitude,
    longitude: stop.longitude,
  }));

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        initialRegion={initialRegion}
      >
        {optimizedRoute.map((stop, index) => (
          <Marker
            key={index}
            coordinate={{
              latitude: stop.latitude,
              longitude: stop.longitude,
            }}
            title={`Stop ${index + 1}`}
            description={stop.name}
          />
        ))}
        <Polyline
          coordinates={coordinates}
          strokeColor="#000"
          strokeWidth={3}
        />
      </MapView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  map: {
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
  },
}); 