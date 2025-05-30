import React from 'react';

let MapContainer, TileLayer, Marker, Polyline;
if (typeof window !== 'undefined') {
  // Only require leaflet on the client
  const leaflet = require('react-leaflet');
  MapContainer = leaflet.MapContainer;
  TileLayer = leaflet.TileLayer;
  Marker = leaflet.Marker;
  Polyline = leaflet.Polyline;
  require('leaflet/dist/leaflet.css');
}

const MapView = ({ center, markers, polyline }) => {
  if (typeof window === 'undefined' || !MapContainer) {
    return null; // Don't render on server
  }
  return (
    <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />
      {markers.map((loc, idx) => (
        <Marker key={idx} position={[loc.latitude, loc.longitude]} />
      ))}
      {polyline.length > 1 && (
        <Polyline positions={polyline} color="#007AFF" />
      )}
    </MapContainer>
  );
};

export default MapView;