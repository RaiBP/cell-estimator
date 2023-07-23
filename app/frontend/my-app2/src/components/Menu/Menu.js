import Button from '../Button/Button'
import React, { useState, useEffect } from 'react'
import axios from 'axios'

import './Menu.css'


const username = 'ami';
const password = '***REMOVED***';

const token = Buffer.from(`${username}:${password}`, 'utf8').toString('base64');

axios.defaults.baseURL = 'https://api.group06.ami.dedyn.io/'
axios.defaults.headers.common['Authorization'] = `Basic ${token}`;

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

function ClassificationMethodsSelector({ onChange, classificationMethods, setClassificationMethods }) {

 useEffect(() => {
   async function fetchData() {
     const response = await axios.get('/get_classification_methods')
     setClassificationMethods(response.data.classification_methods)
   }
   fetchData()
 }, [])

 return (
   <div className="selector-container">
     <label htmlFor='classification' className="selector-label">Choose a classification method:</label>
     <select id='classification' className="selector" onChange={onChange} >
       {classificationMethods.map((method, index) => (
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
  onSegment,
  onImageId,
  onToggleImage,
  onSegmentationMethodChange,
  onClassificationMethodChange,
  onDatasetChange,
  onClassify,
  onSave,
  onDownload,
  isClassified,
  isSegmented,
  onRetrain,
  classificationMethods,
  setClassificationMethods,
  classificationError,
  userDataExists
}) {
  return (
    <div className="menu-container">
      <Button className="menu-button" onClick={onNext}>Next Image</Button>
      <Button className="menu-button" onClick={onPrev}>Previous Image</Button>
      <Button className="menu-button" onClick={onToggleImage}>Toggle Image</Button>
      <Button className="menu-button" onClick={onUndo}>Delete All Polygons</Button>
      <Button className="menu-button" onClick={onReset}>Undo the last Delete</Button>

      <Button className="menu-button" onClick={onSegment}>Segment</Button>
      {isSegmented ? (
  <Button className="menu-button" onClick={onClassify}>
          Classify
  </Button>
) : (
  <Button className="menu-button-disabled" disabled>
            Classify
  </Button>
)} 
      {classificationError && <p className="error-message">Error: Classification failed. Please make sure masks are correct and try again.</p>}
      <form className="selector-container" onSubmit={onImageId}>
        <label className="selector-label">
          Enter a image number:
          <input className="selector" name='image_id' type='number' />
        </label>
        <input className='submit-button' type='submit' value='Submit' />
      </form>
      <SegmentationMethodsSelector onChange={onSegmentationMethodChange}/>
      <ClassificationMethodsSelector onChange={onClassificationMethodChange} classificationMethods={classificationMethods} setClassificationMethods={setClassificationMethods}/>
      <DatasetSelector onChange={onDatasetChange} />
{isClassified ? (
  <Button className="menu-button" onClick={onSave}>
    Save Masks and Labels
  </Button>
) : (
  <Button className="menu-button-disabled" disabled>
    Save Masks and Labels
  </Button>
)}

{userDataExists ? (
  <Button className="menu-button" onClick={onDownload}>
    Download Masks and Labels
  </Button>
) : (
  <Button className="menu-button-disabled" disabled>
    Download Masks and Labels
  </Button>
)}
{userDataExists ? (
  <Button className="menu-button" onClick={onRetrain}>
    Retrain Classification Model
  </Button>
) : (
  <Button className="menu-button-disabled" disabled>
    Retrain Classification Model
  </Button>
)}

    </div>
  )
}

export {
  Menu,
  MenuContainer
}

