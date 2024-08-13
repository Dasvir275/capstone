import React, { useState } from "react";
import { GoogleMap, LoadScript, GroundOverlay } from "@react-google-maps/api";

const MainContent = () => {
  const [imageUrl, setImageUrl] = useState(null);
  const [dischargeOrRiverStage, setDischargeOrRiverStage] = useState("");
  const [riverStageValue, setRiverStageValue] = useState(null);
  const [dischargeValue, setDischargeValue] = useState(null);

  const mapStyles = {
    height: "100%",
    width: "100%",
  };

  const imageBounds = {
    north: 17.276527282781593,
    south: 17.183399247337558,
    east: 81.70444084564792,
    west: 81.60007072846042,
  };


  const handleDischargeOrRiverStageChange = (selectedValue) => {
    setDischargeOrRiverStage(selectedValue);
  };

  const handleRiverStageValueChange = (value) => {
    setRiverStageValue(value);
  };

  const handleDischargeValueChange = (value) => {
    setDischargeValue(value);
  };

  const handleFloodForecast = async () => {
    try {
      let requestData = {
        dischargeOrRiverStage,
      };
  
      if (dischargeOrRiverStage === "discharge") {
        requestData.dischargeValue = parseInt(dischargeValue);
      } else if (dischargeOrRiverStage === "stage") {
        requestData.riverStageValue = parseInt(riverStageValue);
      }
  
      const response = await fetch("http://localhost:5000/flood_forecast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });
  
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
  
      const data = await response.json();
      if (data.success) {
        console.log("Image uploaded successfully!");
        const image_url = data.image_url
        console.log(image_url);
        setImageUrl(image_url);
      } else {
        console.error("Error uploading image:", data.error);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleResetForecast = () => {
    setImageUrl(null);
  };

  return (
    <div className="main_content">
      <div className="input_form">
        
        

        <div className="discharge_value_or_river_stage">
          <label>Enter Discharge Value / River Stage *</label>
          <select
            name="discharge_value_or_river_stage"
            id="discharge_value_or_river_stage"
            value={dischargeOrRiverStage}
            onChange={(e) => handleDischargeOrRiverStageChange(e.target.value)}
          >
            <option value="">Select</option>
            <option value="discharge">Discharge Value</option>
            <option value="stage">River Stage</option>
          </select>

          {dischargeOrRiverStage === "discharge" && (
            <label className="input_discharge_value">
              Discharge Value:
              <input
                type="number"
                value={dischargeValue || ""}
                onChange={(e) => handleDischargeValueChange(e.target.value)}
              />
            </label>
          )}

          {dischargeOrRiverStage === "stage" && (
            <label className="input_river_stage">
              River Stage:
              <input
                type="number"
                value={riverStageValue || ""}
                onChange={(e) => handleRiverStageValueChange(e.target.value)}
              />
            </label>
          )}

          <div className="forecast_buttons">
          <button className="forecast_flood" onClick={handleFloodForecast}>Forecast Flood Indundation</button>
          <button className="reset_forecast" onClick={handleResetForecast}>Reset Forecast</button>
          </div>
        </div>
      </div>

      <div className="map_container">
        <LoadScript googleMapsApiKey="AIzaSyAGmhgPo_KMysxAF3bJTCJrs9RdWJWRhM4">
          <GoogleMap
            mapContainerStyle={mapStyles}
            zoom={13}
            center={{ lat: 17.229963259335577, lng: 81.65222778655493 }}
          >
            {imageUrl && (
              <GroundOverlay
                bounds={imageBounds}
                url={imageUrl}
                opacity={0.2}
              />
            )}
          </GoogleMap>
        </LoadScript>
      </div>
    </div>
  );
};

export default MainContent;