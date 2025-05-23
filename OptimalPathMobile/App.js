import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from './screens/HomeScreen';
import RouteScreen from './screens/RouteScreen';
import DirectionsScreen from './screens/DirectionsScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{ title: 'Optimal Path' }}
        />
        <Stack.Screen 
          name="Route" 
          component={RouteScreen} 
          options={{ title: 'Optimized Route' }}
        />
        <Stack.Screen 
          name="Directions" 
          component={DirectionsScreen} 
          options={{ title: 'Turn-by-Turn Directions' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
} 