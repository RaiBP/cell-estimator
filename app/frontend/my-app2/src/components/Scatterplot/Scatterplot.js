import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell,  } from 'recharts';
import axios from 'axios'
import React, { useState, useEffect } from 'react'
import './Scatterplot.css';

const units = {
      Volume: 'μm³',
      Opacity: '1/μm²',
      DryMassDensity: 'pg/μm³', 
      MaxPhase: 'rad',
      MinPhase: 'rad',
      PhaseVariance: 'rad²',
      PhaseSTDLocalMean: 'rad',
      PhaseSTDLocalVariance: 'rad²',
      PhaseSTDLocalMin: 'rad', 
      PhaseSTDLocalMax: 'rad'    
    };

  const getAxisLabel = (feature) => {
    if (feature in units) {
        const unit = units[feature];
        return `${feature} [${unit}]`;
    } else {
     return feature 
    }
  };

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
      <ScatterChart width={900} height={400}  margin={{ top: 20, right: 0, bottom: 20, left: 30 }}>
        <CartesianGrid />
        <XAxis tick={{fontSize: 20}} dataKey="x" type="number" name={featureX} label={{ value: getAxisLabel(featureX), position: 'insideBottom', offset: -20, fontSize:20 }}/>
        <YAxis tick={{fontSize: 20}} dataKey="y" type="number" name={featureY} label={{ value: getAxisLabel(featureY), angle: -90, position: 'insideLeft', offset: -20, dy:20 , fontSize:20}}/>
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


function ScatterplotFeatureSelector({ onChangeX, onChangeY, featuresList }) {

 return (
   <div className="selector-container-scatter">
     <label htmlFor='scatter' className="selector-label">X-Axis:</label>
     <select id='scatter-x-feature' className="selector" onChange={onChangeX} >
       {featuresList.map((feature, index) => (
         <option key={index} value={feature}>
           {feature}
         </option>
       ))}
     </select>
     <label htmlFor='scatter' className="selector-label">Y-Axis:</label>
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


function Scatterplot({
  featuresList,
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
      <ScatterplotFeatureSelector featuresList={featuresList} onChangeX={onFeatureChangeX} onChangeY={onFeatureChangeY}/>
      <ScatterplotComponent scatterplotDataX={scatterDataX} scatterplotDataY={scatterDataY} scatterplotDataColor={scatterDataColor} featureX={featureX} featureY={featureY} onPointHover={onPointHover} key="Scatterplot Plot"/>
    </div>
  )
}

export {
  Scatterplot
}
