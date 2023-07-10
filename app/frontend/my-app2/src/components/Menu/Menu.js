import Button from '../Button/Button'
import React, { useState, useEffect } from 'react'
import axios from 'axios'

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
    <div>
      <label for='dataset-selector'>Choose a dataset: </label>
      <select id='dataset-selector' onChange={onChange}>
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
    <div>
      <label for='segmentation'>Choose a segmentation method:</label>
      <select id='segmentation' onChange={onChange}>
        {segmentationMethods.map((method, index) => (
          <option key={index} value={method}>
            {method}
          </option>
        ))}
      </select>
    </div>
  )
}

function Menu({
  onReset,
  onUndo,
  onSave,
  onNext,
  onPrev,
  onImageId,
  onToggleImage,
  onSegmentationMethodChange,
  onDatasetChange,
}) {
  return (
    <div
      style={{
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        width: '10%',
        background: '#f0f0f0',
        padding: '0px',
      }}
    >
      <Button onClick={onReset}>Reset</Button>
      <Button onClick={onUndo}>Undo</Button>
      <Button onClick={onSave}>Save</Button>
      <Button onClick={onNext}>Next Image</Button>
      <Button onClick={onPrev}>Previous Image</Button>
      <Button onClick={onToggleImage}>Toggle Image</Button>
      <form onSubmit={onImageId}>
        <label>
          Enter a number between 1 and 1000:
          <input name='image_id' type='number' />
        </label>
        <input type='submit' value='Submit' />
      </form>
      <SegmentationMethodsSelector onChange={onSegmentationMethodChange} />
      <DatasetSelector onChange={onDatasetChange} />
    </div>
  )
}

export default Menu
