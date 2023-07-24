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

// Helper function to format tick values with a specific number of decimals
const formatTickValue = (value, decimals) => {
  return Number(value).toFixed(decimals);
};

const ScatterplotComponent = ({ 
  scatterplotDataX, 
  scatterplotDataY, 
  scatterplotDataColor, 
  featureX, 
  featureY, 
  onPointHover, 
  scatterTrainingDataX, 
  scatterTrainingDataY, 
  scatterTrainingDataColor 
}) => {
  if (!scatterplotDataX || !scatterplotDataY) {
    return <div>No Data</div>;  
  }

  // Combine the scatterplot data and training data if they exist
  const data = scatterplotDataX.map((x, index) => ({
    x: x,
    y: scatterplotDataY[index],
    color: scatterplotDataColor[index],
    isTrainingData: false
  }));

const trainingData = scatterTrainingDataX && scatterTrainingDataY && scatterTrainingDataColor
    ? scatterTrainingDataX.map((x, index) => ({
        x: x,
        y: scatterTrainingDataY[index],
        color: scatterTrainingDataColor[index],
        isTrainingData: true
      }))
    : [];

  // Concatenate the training data with the scatterplot data
  data.push(...trainingData);

  const handlePointMouseEnter = (point, index) => {
    console.log(index)
    onPointHover(index);
  };

  const handlePointMouseLeave = () => {
    onPointHover(null);
  };

  // Custom tooltip content with smaller font size
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="label" style={{ fontSize: '18px' }}>
            X: {payload[0].payload.x}
          </p>
          <p className="label" style={{ fontSize: '18px' }}>
            Y: {payload[0].payload.y}
          </p>
        </div>
      );
    }
    return null;
  };

  // Find the minimum and maximum values in the data to set the axis domain
  const minX = Math.min(...data.map(entry => entry.x));
  const maxX = Math.max(...data.map(entry => entry.x));
  const minY = Math.min(...data.map(entry => entry.y));
  const maxY = Math.max(...data.map(entry => entry.y));

  // Calculate the margin as 10% of the data range
  const xMargin = (maxX - minX) * 0.1;
  const yMargin = (maxY - minY) * 0.1;

  // Adjust the axis domain with the margins
  const xAxisDomain = [minX - xMargin, maxX + xMargin];
  const yAxisDomain = [minY - yMargin, maxY + yMargin];

 // Define the number of decimals you want for the ticks
  const xAxisDecimals = 2;
  const yAxisDecimals = 2;

  // Custom tick formatter functions
  const formatXAxisTick = (tick) => formatTickValue(tick, xAxisDecimals);
  const formatYAxisTick = (tick) => formatTickValue(tick, yAxisDecimals);

  // Define the number of ticks you want on each axis
  const xAxisTickCount = 5; // You can adjust this value as needed
  const yAxisTickCount = 5; // You can adjust this value as needed

  // Calculate the tick interval for each axis
  const xAxisTickInterval = (maxX - minX) / (xAxisTickCount - 1);
  const yAxisTickInterval = (maxY - minY) / (yAxisTickCount - 1);

  return (
    <div className="scatter-component">
      {/* Render your scatterplot using the retrieved data */}
      <ScatterChart width={1600} height={800}  margin={{ top: 20, right: 0, bottom: 25, left: 145 }}>
        <CartesianGrid />
        <XAxis tick={{fontSize: 20}} dataKey="x" type="number" name={featureX} label={{ value: getAxisLabel(featureX), position: 'insideBottom', offset: -20, fontSize:20}} domain={xAxisDomain} tickFormatter={formatXAxisTick} ticks={[...Array(xAxisTickCount)].map((_, index) => minX + xAxisTickInterval * index)}/>
        <YAxis tick={{fontSize: 20}} dataKey="y" type="number" name={featureY} label={{ value: getAxisLabel(featureY), angle: -90, position: 'insideLeft', offset: -35, dy:100 , fontSize:20}} domain={yAxisDomain} tickFormatter={formatYAxisTick} ticks={[...Array(yAxisTickCount)].map((_, index) => minY + yAxisTickInterval * index)} // Manually set tick values
        />
        <Tooltip content={<CustomTooltip />} />

        {/* Render the training data points */}
        <Scatter data={trainingData} shape="circle" fillOpacity={0.4}>
          {trainingData.map((entry, index) => (
            <Cell
              key={`cell-training-${index}`}
              fill={entry.color}
              strokeWidth={1}
              size={30}
            />
          ))}
        </Scatter>

        {/* Render the regular data points */}
        <Scatter data={data.filter(entry => !entry.isTrainingData)} shape="circle" onMouseEnter={handlePointMouseEnter}
          onMouseLeave={handlePointMouseLeave}>
          {data.map((entry, index) => (
            !entry.isTrainingData && (
              <Cell
                key={`cell-${index}`}
                fill={entry.color}
              stroke='#000'
                strokeWidth={2.5}
                size={160}
              />
            )
          ))}
        </Scatter>
      </ScatterChart>
    </div>
  );
};

export default ScatterplotComponent;


function ScatterplotFeatureSelector({ onChangeX, onChangeY, featuresList, handleTrainingDataToggle, showTrainingData}) {

 return (
   <div className="selector-container-scatter">
     <label htmlFor='scatter' className="selector-label-scatter">X-Axis:</label>
     <select id='scatter-x-feature' className="selector-scatter" onChange={onChangeX} >
       {featuresList.map((feature, index) => (
         <option key={index} value={feature}>
           {feature}
         </option>
       ))}
     </select>
     <label htmlFor='scatter' className="selector-label-scatter">Y-Axis:</label>
     <select id='scatter-y-feature' className="selector-scatter" onChange={onChangeY} >
       {featuresList.map((feature, index) => (
         <option key={index} value={feature}>
           {feature}
         </option>
       ))}
     </select>
 <div className="plot-training-data">
        <input
          type="checkbox"
          id="show-training-data"
          checked={showTrainingData}
          onChange={handleTrainingDataToggle}
        />
        <label htmlFor="show-training-data">Show Training Data</label>
      </div>
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
  scatterTrainingDataX,
  scatterTrainingDataY,
  scatterTrainingDataColor,
  onFeatureChangeX,
  onFeatureChangeY,
  onPointHover,
  handleTrainingDataToggle,
  showTrainingData
}) {
  return (
    <div className="scatter-container">
      <ScatterplotFeatureSelector featuresList={featuresList} onChangeX={onFeatureChangeX} onChangeY={onFeatureChangeY} handleTrainingDataToggle={handleTrainingDataToggle} showTrainingData={showTrainingData}/>
      <ScatterplotComponent scatterplotDataX={scatterDataX} scatterplotDataY={scatterDataY} scatterplotDataColor={scatterDataColor} featureX={featureX} featureY={featureY} onPointHover={onPointHover} scatterTrainingDataX={scatterTrainingDataX} scatterTrainingDataY={scatterTrainingDataY} scatterTrainingDataColor={scatterTrainingDataColor} key="Scatterplot Plot"/>
    </div>
  )
}

export {
  Scatterplot
}
