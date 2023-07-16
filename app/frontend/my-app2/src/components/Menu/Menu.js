import Button from '../Button/Button'
import React, { useState, useEffect } from 'react'
import axios from 'axios'

import './Menu.css'

function DatasetSelector({ onChange }) {
  const [datasets, setDatasets] = useState([])

  useEffect(() => {
    async function fetchData() {
      const response = await axios.get('/datasets')
      setDatasets(response.data.datasets)
    }
    fetchData()
  }, [])

  return (
    <div className="selector-container">
      <label htmlFor='dataset-selector' className="selector-label">Choose a dataset: </label>
      <select id='dataset-selector' className="selector" onChange={onChange}>
        {datasets.map((method, index) => (
          <option key={index} value={method}>
            {method}
          </option>
        ))}
      </select>
    </div>
  )
}

function SegmentationMethodsSelector({ onChange }) {
  const [segmentationMethods, setSegmentationMethods] = useState([])

  useEffect(() => {
    async function fetchData() {
      const response = await axios.get('/get_segmentation_methods')
      setSegmentationMethods(response.data.segmentation_methods)
    }
    fetchData()
  }, [])

  return (
    <div className="selector-container">
      <label htmlFor='segmentation' className="selector-label">Choose a segmentation method:</label>
      <select id='segmentation' className="selector" onChange={onChange}>
        {segmentationMethods.map((method, index) => (
          <option key={index} value={method}>
            {method}
          </option>
        ))}
      </select>
    </div>
  )
}

const MenuContainer = ({ children }) => {
  return <div className="menu-container">{children}</div>
}

function Menu({
  onReset,
  onUndo,
  onNext,
  onPrev,
  onImageId,
  onToggleImage,
  onSegmentationMethodChange,
  onDatasetChange,
  onClassification
}) {
  return (
    <div className="menu-container">
      <Button className="menu-button" onClick={onNext}>Next Image</Button>
      <Button className="menu-button" onClick={onPrev}>Previous Image</Button>
      <Button className="menu-button" onClick={onToggleImage}>Toggle Image</Button>
      <Button className="menu-button" onClick={onUndo}>Delete All Polygons</Button>
      <Button className="menu-button" onClick={onReset}>Undo the last Delete</Button>
      
      <form className="selector-container" onSubmit={onImageId}>
        <label className="selector-label">
          Enter a number between 1 and 1000:
          <input className="selector" name='image_id' type='number' />
        </label>
        <input type='submit' value='Submit' />
      </form>
      <SegmentationMethodsSelector onChange={onSegmentationMethodChange} />
      <DatasetSelector onChange={onDatasetChange} />
      <Button className="menu-button" onClick={onClassification}>Classify</Button>
    </div>
  )
}

export {
  Menu,
  MenuContainer
}

