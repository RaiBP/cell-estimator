import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell,  } from 'recharts';
import axios from 'axios'
import React, { useState, useEffect } from 'react'
import './Scatterplot.css';



const ScatterplotComponent = ({ scatterplotDataX, scatterplotDataY, scatterplotDataColor, featureX, featureY, onPointHover }) => {
  if (!scatterplotDataX || !scatterplotDataY) {
    return <div>No Data</div>;
  }

  const data = scatterplotDataX.map((x, index) => ({
    x: x,
    y: scatterplotDataY[index],
    color: scatterplotDataColor[index]
  }));

  const handlePointMouseEnter = (point, index) => {
    console.log(index)
    onPointHover(index);
  };

  const handlePointMouseLeave = () => {
    onPointHover(null);
  };

  return (
    <div className="scatter-component">
      {/* Render your scatterplot using the retrieved data */}
      <ScatterChart width={1000} height={500}>
        <CartesianGrid />
        <XAxis dataKey="x" type="number" name={featureX} label={{ value: featureX, position: 'insideBottom', offset: -5 }}/>
        <YAxis dataKey="y" type="number" name={featureY} label={{ value: featureY, angle: -90, position: 'insideLeft', offset: 5 }}/>
        <Tooltip />
        <Scatter
          data={data}
          fill="#8884d8"
          onMouseEnter={handlePointMouseEnter}
          onMouseLeave={handlePointMouseLeave}
>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Scatter>
      </ScatterChart>
    </div>
  );
};


function ScatterplotFeatureSelector({ onChangeX, onChangeY }) {
  const [featuresList, setFeaturesList] = useState([])

  useEffect(() => {
    async function fetchFeaturesList() {
      const response = await axios.get('/available_features_names')
      setFeaturesList(response.data.features)
    }
    fetchFeaturesList()
  }, [])

 return (
   <div className="selector-container-scatter">
     <label htmlFor='scatter' className="selector-label">Choose feature for x-axis:</label>
     <select id='scatter-x-feature' className="selector" onChange={onChangeX} >
       {featuresList.map((feature, index) => (
         <option key={index} value={feature}>
           {feature}
         </option>
       ))}
     </select>
     <label htmlFor='scatter' className="selector-label">Choose feature for y-axis:</label>
     <select id='scatter-y-feature' className="selector" onChange={onChangeY} >
       {featuresList.map((feature, index) => (
         <option key={index} value={feature}>
           {feature}
         </option>
       ))}
     </select>
   </div>
 )
}


const ScatterplotContainer = ({ children }) => {
  return <div className="scatter-container" key="Scatterplot wrapper">{children}</div>
}

function Scatterplot({
  featureX,
  featureY,
  scatterDataX,
  scatterDataY,
  scatterDataColor,
  onFeatureChangeX,
  onFeatureChangeY,
  onPointHover
}) {
  return (
    <div className="scatter-container">
      <ScatterplotFeatureSelector onChangeX={onFeatureChangeX} onChangeY={onFeatureChangeY}/>
      <ScatterplotComponent scatterplotDataX={scatterDataX} scatterplotDataY={scatterDataY} scatterplotDataColor={scatterDataColor} featureX={featureX} featureY={featureY} onPointHover={onPointHover} key="Scatterplot Plot"/>
    </div>
  )
}

export {
  Scatterplot
}
