import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView, Alert } from 'react-native';
import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Change this to your backend URL

export default function HomeScreen({ navigation }) {
  const [stops, setStops] = useState([]);
  const [currentAddress, setCurrentAddress] = useState('');

  const addStop = async () => {
    if (!currentAddress.trim()) {
      Alert.alert('Error', 'Please enter an address');
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/geocode`, {
        address: currentAddress
      });

      setStops([...stops, {
        name: currentAddress,
        latitude: response.data.latitude,
        longitude: response.data.longitude
      }]);
      setCurrentAddress('');
    } catch (error) {
      Alert.alert('Error', 'Could not geocode address');
    }
  };

  const optimizeRoute = async () => {
    if (stops.length < 2) {
      Alert.alert('Error', 'Please add at least 2 stops');
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/optimize_route`, {
        stops: stops
      });

      navigation.navigate('Route', {
        optimizedRoute: response.data.optimized_route
      });
    } catch (error) {
      Alert.alert('Error', 'Could not optimize route');
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollView}>
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Enter address"
            value={currentAddress}
            onChangeText={setCurrentAddress}
          />
          <TouchableOpacity style={styles.button} onPress={addStop}>
            <Text style={styles.buttonText}>Add Stop</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.stopsContainer}>
          <Text style={styles.title}>Stops:</Text>
          {stops.map((stop, index) => (
            <Text key={index} style={styles.stopText}>
              {index + 1}. {stop.name}
            </Text>
          ))}
        </View>

        <TouchableOpacity 
          style={[styles.button, styles.optimizeButton]} 
          onPress={optimizeRoute}
        >
          <Text style={styles.buttonText}>Optimize Route</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollView: {
    flex: 1,
    padding: 20,
  },
  inputContainer: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 10,
    marginRight: 10,
    borderRadius: 5,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 5,
    justifyContent: 'center',
  },
  buttonText: {
    color: '#fff',
    textAlign: 'center',
  },
  stopsContainer: {
    marginBottom: 20,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  stopText: {
    fontSize: 16,
    marginBottom: 5,
  },
  optimizeButton: {
    backgroundColor: '#34C759',
    marginTop: 20,
  },
}); 