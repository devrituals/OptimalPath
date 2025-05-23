import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Change this to your backend URL

export default function DirectionsScreen({ route }) {
  const { optimizedRoute } = route.params;
  const [directions, setDirections] = useState([]);

  useEffect(() => {
    const fetchDirections = async () => {
      try {
        const directionsPromises = [];
        for (let i = 0; i < optimizedRoute.length - 1; i++) {
          const start = optimizedRoute[i];
          const end = optimizedRoute[i + 1];
          
          const response = await axios.post(`${API_URL}/directions`, {
            start_lat: start.latitude,
            start_lon: start.longitude,
            end_lat: end.latitude,
            end_lon: end.longitude,
          });
          
          directionsPromises.push({
            from: start.name,
            to: end.name,
            steps: response.data.steps,
            total_distance: response.data.total_distance,
            total_duration: response.data.total_duration,
          });
        }
        
        setDirections(directionsPromises);
      } catch (error) {
        console.error('Error fetching directions:', error);
      }
    };

    fetchDirections();
  }, [optimizedRoute]);

  return (
    <ScrollView style={styles.container}>
      {directions.map((segment, index) => (
        <View key={index} style={styles.segment}>
          <Text style={styles.segmentTitle}>
            Segment {index + 1}: {segment.from} → {segment.to}
          </Text>
          <Text style={styles.segmentInfo}>
            Distance: {(segment.total_distance / 1000).toFixed(2)} km
          </Text>
          {segment.steps.map((step, stepIndex) => (
            <View key={stepIndex} style={styles.step}>
              <Text style={styles.stepNumber}>{stepIndex + 1}</Text>
              <Text style={styles.stepText}>{step.instruction}</Text>
            </View>
          ))}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  segment: {
    marginBottom: 20,
    padding: 15,
    backgroundColor: '#f8f8f8',
    borderRadius: 10,
  },
  segmentTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  segmentInfo: {
    fontSize: 16,
    color: '#666',
    marginBottom: 10,
  },
  step: {
    flexDirection: 'row',
    marginBottom: 10,
    alignItems: 'center',
  },
  stepNumber: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#007AFF',
    color: '#fff',
    textAlign: 'center',
    lineHeight: 30,
    marginRight: 10,
  },
  stepText: {
    flex: 1,
    fontSize: 16,
  },
}); 