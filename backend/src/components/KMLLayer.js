import React, { useState } from 'react';
import GoogleMap from 'react-google-maps';
import KMLLayer from 'react-google-maps/components/KMLLayer';

const KMLOverlay = () => {
  const [kmlUrl, setKmlUrl] = useState('');

  return (
    <GoogleMap
      defaultZoom={13}
      defaultCenter={{
        lat: 37.7833,
        lng: -122.4167,
      }}
    >
      <KMLLayer url={kmlUrl} />
    </GoogleMap>
  );
};

export default KMLOverlay;