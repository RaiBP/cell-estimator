import React, { useEffect, useState } from 'react'
import { Stage, Layer, Line, Image, Circle, Group } from 'react-konva'
import axios from 'axios'
import { Menu, MenuContainer } from './components/Menu/Menu'
import { PopupMenu } from './components/PopupMenu/PopupMenu'
import { ExplainMenu } from './components/ExplainMenu/ExplainMenu'
import { Scatterplot } from './components/Scatterplot/Scatterplot';
import { Legend } from './components/Legend/Legend'
import { v4 as uuidv4 } from 'uuid'

import './App.css'

axios.defaults.baseURL = 'http://localhost:8000'

const stageDimensions = {
  width: 1000,
  height: 800,
}

const StageContainer = ({ children }) => {
  const style = {
    display: 'flex',
    justifyContent: 'left',
    alignItems: 'center',
    flex: 1,
    width: '100%',
    height: '100%',
    paddingLeft: '1px',
    paddingTop: '50px',
    overflow: 'auto',
    backgroundColor: "#351C75",
  }

  return <div style={style}>{children}</div>
}

async function setNewImage(imageId, imageType, callback) {
  const response = await fetch('http://localhost:8000/set_image', {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image_id: imageId, image_type: imageType }),
  })
  const response_json = await response.json()
  callback(response_json)
  return response_json
}

async function segmentCurrentImage(callback) {
  const response = await axios.get('/segment')
  const polygons = response.data.polygons
  callback(polygons)
  return polygons
}

function divideElements(objectOfArrays) {
  const width = stageDimensions.width
  const height = stageDimensions.height
  const data = {}

  for (let key in objectOfArrays) {
    if (objectOfArrays.hasOwnProperty(key)) {
      data[key] = objectOfArrays[key].map((element) => {
        return {
          x: element.x / width,
          y: element.y / height,
        }
      })
    }
  }

  const transformedData = []

  // Iterate through the original data
  for (const key in data) {
    if (data.hasOwnProperty(key)) {
      const points = data[key]
      const transformedPoints = []

      // Extract x and y values for each point
      for (const point of points) {
        const { x, y } = point

        // Create a new object with the desired format
        transformedPoints.push(x, y)
      }

      // Push the new object into the transformed data array
      transformedData.push({ points: transformedPoints })
    }
  }
  return transformedData
}

function LoadingSpinner() {
  return (
    <div className='loading-indicator'>
      <div className='spinner'></div>
    </div>
  )
}

const AnnotationArea = () => {
  const style = {
    display: 'flex',
    alignItems: 'flex-start', // align items vertically
    justifyContent: 'flex-start', // align items horizontally
    height: '100%', // 100% of the viewport height
    width: '100%', // 100% of the viewport height
    backgroundColor: '#351C75',
  }

  // Image management
  const [amplitudeImage, setAmplitudeImage] = useState(null)
  const [phaseImage, setPhaseImage] = useState(null)
  const [showAmplitudeImage, setShowAmplitudeImage] = useState(true) // 0 for amplitude, 1 for phase
  const [image, setImage] = useState(null)
  const [imageId, setImageId] = useState(0)
  const [currentDataset, setCurrentDataset] = useState(null)
  const img = new window.Image()

  // Polygon management
  const [polygons, setPolygons] = useState([])
  const [currentPolygon, setCurrentPolygon] = useState([])
  const currentPolygonRef = React.useRef(currentPolygon)
  const [nextPoint, setNextPoint] = useState(null)
  const [deletedPolygons, setDeletedPolygons] = useState([])
  const [numberOfDeletedPolygons, setNumberOfDeletedPolygons] = useState([])
  const [isLoading, setIsLoading] = useState(false);
  const [isSegmented, setIsSegmented] = useState(false);

  // Preview line management
  const [previewLine, setPreviewLine] = useState(null)
  const [isClassified, setIsClassified] = useState(false)
 const [classificationMethods, setClassificationMethods] = useState([])
  const [scatterplotDataX, setScatterplotDataX] = useState(null);
  const [scatterplotDataY, setScatterplotDataY] = useState(null);
  const [deletedXData, setDeletedXData] = useState([]);
  const [deletedYData, setDeletedYData] = useState([]);
  const [deletedColorData, setDeletedColorData] = useState([]);
  const [scatterplotDataColor, setScatterplotDataColor] = useState(null);
  const [featureXAxis, setFeatureXAxis] = useState("Volume");
  const [featureYAxis, setFeatureYAxis] = useState("Volume");
const [activePoint, setActivePoint] = useState(null);
  const [classificationError, setClassificationError] = useState(false);
  const [userDataExists, setUserDataExists] = useState(false);

  // Context Menu for Polygon-editing
  const [contextMenu, setContextMenu] = useState({
    visible: false,
    x: 0,
    y: 0,
    polygonID: -1,
  })
  const [explainMenu, setExplainMenu] = useState({
    visible: false,
    polygonID: -1,
  })
  const [availableFeaturesNames, setAvailableFeaturesNames] = useState([])

  const [imageUrl, setImageUrl] = useState('');
  // Component management
  const stageRef = React.useRef()

  // Most uncertain
  const [mostUncertain,setMostUncertain]= useState([])

  async function retrainModel() {
    setIsLoading(true)
    const response = await axios.get('/retrain_model')
    console.log(response)
    const methods = await axios.get('/get_classification_methods')
    setIsLoading(false)
    console.log(methods.data)
    setClassificationMethods(methods.data.classification_methods)
  }

  async function classifyCurrentImage(callback) {
    const masks = divideElements(polygons)
    setIsLoading(true)
    const response = await fetch('http://localhost:8000/classify', {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ polygons: masks, use_backend_masks: false }),
    })
    const predictions = await response.json()
    setIsLoading(false)
    callback(predictions)
    console.log(predictions)
    const mu=find_max_entropy(predictions)
    setMostUncertain(mu)
    return predictions
  }

  async function checkUserData() {
    setIsLoading(true)
    const response = await axios.get('/user_data_exists')
    setUserDataExists(response.data.value)
    setIsLoading(false)
  }

  function find_max_entropy(objects){
    // Key to compare values against
    const keyToCompare = 'LabelsEntropy';
    let threshold = 1.3;
    let maxObject = [];

    // Iterate through each object
    for (const obj of objects) {
    // Access the value of the specified key
    const value = obj.features[keyToCompare];

    // Compare the value with the current maximum value
    if (value > threshold) {
    // Update the maximum value and corresponding object
    maxObject.push(obj);
    }}

    // maxObject now holds the object with the largest value for the specified key
    const maskIDs = maxObject.map(obj => obj.features.MaskID);
    return maskIDs;
}

  function getColorByClassId(classId) {
    switch (classId) {
      case 'rbc':
        return '#ff0000'
      case 'wbc':
        return '#ffffff'
      case 'plt':
        return '#0000ff'
      case 'agg':
        return '#00ff00'
      case 'oof':
        return '#ffff00'
    }
  }

  function getClassIdByColor(color) {
    console.log(color)
    switch (color) {
      case '#ff0000':
        return 'rbc'
      case '#ffffff':
        return 'wbc'
      case '#0000ff':
        return 'plt'
      case '#00ff00':
        return 'agg'
      case '#ffff00':
        return 'oof'
    }
  }

  function getClassIdFromPolygon(polygon) {
    return getClassIdByColor(polygon[0].color)
  }

  async function saveCurrentMaskAndLabels(labels) {
    console.log(labels)
    try {
      setIsLoading(true)
      const response = await axios.post('/save_masks_and_labels', labels)
      setIsLoading(false)
      console.log(response.data)
    } catch (error) {
      console.error('Error saving labels:', error)
      setIsLoading(false)
    }
  }

  async function downloadMasksAndLabels() {
    try {
      setIsLoading(true)
      const response = await axios.get('/download_masks_and_labels', {
        responseType: 'blob',
      })

      setIsLoading(false)

      // Create a timestamp
      const timestamp = new Date().toISOString().replace(/:/g, '-')
      const filename = `masks_and_labels_${timestamp}.pre`

      // Create a download link for the user
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      // Clean up the URL object
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error downloading masks and labels:', error)
      // Optionally handle error cases
    }
  }

  function segmentCallback(receivedPolygons) {
    setClassificationError(false);
    const transformedPolygons = []

    if (receivedPolygons.length !== 0) {
      receivedPolygons.forEach((receivedPolygons, index) => {
        const currentPolygon = []
        for (let i = 0; i < receivedPolygons.points.length; i += 8) {
          currentPolygon.push({
            x: receivedPolygons.points[i] * stageDimensions.width,
            y: receivedPolygons.points[i + 1] * stageDimensions.height,
            color: '#ffa500',
            id: uuidv4(),
          })
        }
        transformedPolygons.push(currentPolygon)
      })
    }
    setPolygons(transformedPolygons)
    setIsClassified(false)
    setScatterplotDataX(null)
    setScatterplotDataY(null)
    setScatterplotDataColor(null)
    setDeletedPolygons([])
    setNumberOfDeletedPolygons([])
    setDeletedXData([])
    setDeletedYData([])
    setDeletedColorData([])
  }

  async function classifyCallback(labels) {
    setClassificationError(false);
    const newFeaturesScatterDataX = []
    const newFeaturesScatterDataY = []
    const newFeaturesColor = []

    if (labels.length !== 0) { 
      const transformedPolygons = polygons.map((polygon, index) => {
        const classId = labels[index]['class_id']

        console.log(`Polygon ${index + 1} - classId: ${classId}`)

        const color = getColorByClassId(classId)

        newFeaturesScatterDataX.push(labels[index]['features'][featureXAxis])
        newFeaturesScatterDataY.push(labels[index]['features'][featureYAxis])
        newFeaturesColor.push(color)

        return polygon.map((point) => ({
          ...point,
          color: color,
        }))
      })

      setScatterplotDataX(newFeaturesScatterDataX)
      setScatterplotDataY(newFeaturesScatterDataY)
      setScatterplotDataColor(newFeaturesColor)

      setPolygons(transformedPolygons)
      setIsClassified(true)

      if (availableFeaturesNames.length == 0) {
        fetchAvailableFeaturesNames()
      }
  }
  else {
      if (polygons.length !== 0) {
        // if this is the case, there has been an error with the classification
        setClassificationError(true)
      }
    }

    console.log(`Available feature names: ${availableFeaturesNames.length}`)
  }


  async function setImageCallback(response_json) {
    // This is a callback function that is called when the image is fetched
    // Its only purpose is to set the image state variables 
    setClassificationError(false);

    setAmplitudeImage(
      `data:image/jpeg;base64,${response_json.amplitude_img_data}`
    )
    setPhaseImage(`data:image/jpeg;base64,${response_json.phase_img_data}`)

    const transformedPolygons = []
    const newDataX = []
    const newDataY = []
    const newDataColor = []

    const polygonsWithPredictions = response_json.predictions
    if (polygonsWithPredictions.length !== 0) {
      polygonsWithPredictions.forEach((polygonWithPrediction, index) => {
        const currentPolygon = []
        for (
          let i = 0;
          i < polygonWithPrediction.polygon.points.length;
          i += 8
        ) {
          currentPolygon.push({
            x: polygonWithPrediction.polygon.points[i] * stageDimensions.width,
            y:
              polygonWithPrediction.polygon.points[i + 1] *
              stageDimensions.height,
            color: getColorByClassId(polygonWithPrediction.class_id),
            id: uuidv4(),
          })
        }
        transformedPolygons.push(currentPolygon)

        newDataX.push(polygonWithPrediction.features[featureXAxis])
        newDataY.push(polygonWithPrediction.features[featureYAxis])
        newDataColor.push(getColorByClassId(polygonWithPrediction.class_id))
        
      })      

      setScatterplotDataX(newDataX)
      setScatterplotDataY(newDataY)
      setScatterplotDataColor(newDataColor)
      setIsClassified(true)

      if (availableFeaturesNames.length == 0) {
        fetchAvailableFeaturesNames()
      }
    } else {
      setIsClassified(false)

      setScatterplotDataX(null)
      setScatterplotDataY(null)
      setScatterplotDataColor(null)
    }
    setPolygons(transformedPolygons) 
  }

 const onPointHover = (index) => {
    setActivePoint(index);
  };

// Hook for checking if there are any drawn polygons
useEffect(() => {
  setIsSegmented(polygons.length !== 0);
}, [polygons]);

  useEffect(() => {
    // Call the function when the app opens
    checkUserData();
  }, []);

useEffect(() => {
    console.log(`Current Image ID: ${imageId}`);
}, [imageId]);

// Hook for showing amplitude or phase image
useEffect(() => {
  if (showAmplitudeImage) {
    img.src = amplitudeImage
  } else {
    img.src = phaseImage
  }
  img.onload = () => {
    setImage(img)
  }
}, [showAmplitudeImage, amplitudeImage, phaseImage])

useEffect(() => {
  const image_type = showAmplitudeImage ? 0 : 1

  const setNewImageAsync = async () => {
    setIsLoading(true)
    await setNewImage(imageId, image_type, setImageCallback)
    setIsLoading(false)
  }

  setNewImageAsync()
}, [imageId, showAmplitudeImage, currentDataset])

// Hook for keeping track of lines
useEffect(() => {
  currentPolygonRef.current = currentPolygon
}, [currentPolygon])

// Hook for registering keydown events -- happens only when component is mounted
useEffect(() => {
  // Handling keydown events -- registering callback
  const handleKeyDown = (event) => {
    if (event.key === 'r') {
      deleteall()
    } else if (event.key === 'z' && event.ctrlKey) {
    } else if (event.key === 'Escape') {
      finishPolygon()
    } else if (event.key === 'ArrowRight') {
      nextImage()
    } else if (event.key === 'ArrowLeft') {
      prevImage()
    } else if (event.key === 't') {
      toggleImage()
    }
  }

  window.addEventListener('keydown', handleKeyDown)
  return () => {
    window.removeEventListener('keydown', handleKeyDown)
  }
}, [])

  const handleClick = (e) => {
    if (e.evt.button === 0) {
      const mousePos = stageRef.current.getStage().getPointerPosition()
      setCurrentPolygon([
        ...currentPolygon,
        { x: mousePos.x, y: mousePos.y, color:'#ffa500', id: uuidv4() },
      ])
      console.log(currentPolygon)
    } else if (e.evt.button === 2) {
      e.evt.preventDefault()
    }
  }

  const handleMouseMove = (e) => {
    const mousePos = stageRef.current.getStage().getPointerPosition()
    setNextPoint({ x: mousePos.x, y: mousePos.y })
  }

  const finishPolygon = () => {
    if (currentPolygonRef.current.length > 1) {
      setPolygons((prevPolygons) => [
        ...prevPolygons,
        currentPolygonRef.current,
      ])

      setDeletedXData([])
      setDeletedYData([])
      setDeletedColorData([])

      setScatterplotDataX(null)
      setScatterplotDataY(null)
      setScatterplotDataColor(null)

    }
    setCurrentPolygon([])
    setNextPoint(null)
  }

  const nextImage = () => {
    setImageId((prevId) => prevId + 1)
    setDeletedPolygons([])
    setNumberOfDeletedPolygons([])
    setMostUncertain(null)
  }

  const prevImage = () => {
    setImageId((prevId) => prevId - 1)
    setDeletedPolygons([])
    setNumberOfDeletedPolygons([])
    setMostUncertain(null)
  }

  const handleButtonClick = (e) => {
    e.preventDefault()

    const newImageId = e.target.image_id.value
    // Validate the number
    if (newImageId >= 0 && newImageId <= 9999) {
      setImageId(newImageId)
      // Perform your desired action with the valid number
      console.log('Valid number:', newImageId)
    } else {
      console.log('Invalid number:', newImageId)
    }
  }

  const segment = async () => {
    setIsLoading(true)
    await segmentCurrentImage(segmentCallback)
    setIsLoading(false)
  }

  const classify = () => {
    classifyCurrentImage(classifyCallback)
  }

  const retrain = () => {
    retrainModel()
  }

  const download = () => {
    downloadMasksAndLabels()
  }

  const saveMasksAndLabels = () => {
    const extractedLabels = polygons.map((polygon) => {
      return getClassIdFromPolygon(polygon)
    })

    saveCurrentMaskAndLabels(extractedLabels)
  }

  const toggleImage = () => {
    setShowAmplitudeImage((prev) => !prev)
  }

  function undoLast() {
    let lastNumber = numberOfDeletedPolygons.splice(-1, 1)
    lastNumber = lastNumber[0]
    let recoveredPolygon = []
    let recoveredDataX = [];
    let recoveredDataY = [];
    let recoveredDataColor = [];

    if (lastNumber === 1) {
      recoveredPolygon = deletedPolygons.splice(-1, 1)
      recoveredDataX = deletedXData.splice(-1, 1) 
      recoveredDataY = deletedYData.splice(-1, 1)
      recoveredDataColor = deletedColorData.splice(-1, 1)
    } else if (lastNumber > 1) {
      recoveredPolygon = deletedPolygons.splice(-lastNumber, lastNumber)
      recoveredDataX = deletedXData.splice(-lastNumber, lastNumber) 
      recoveredDataY = deletedYData.splice(-lastNumber, lastNumber)
      recoveredDataColor = deletedColorData.splice(-lastNumber, lastNumber)
    }

    polygons.push(...recoveredPolygon)
    if (scatterplotDataX === null) {
      setScatterplotDataX(recoveredDataX)
    } else {
      scatterplotDataX.push(...recoveredDataX)
    }

    if (scatterplotDataY === null) {
      setScatterplotDataY(recoveredDataY)
    } else {
      scatterplotDataY.push(...recoveredDataY)
    }

    if (scatterplotDataColor === null) {
      setScatterplotDataColor(recoveredDataColor)
    } else {
      scatterplotDataColor.push(...recoveredDataColor)
    }
  }

  function deleteall() {
    setNumberOfDeletedPolygons([...numberOfDeletedPolygons, polygons.length])
    setDeletedPolygons([...deletedPolygons, ...polygons])
    setPolygons([])

    setDeletedXData([...deletedXData, ...scatterplotDataX])
    setDeletedYData([...deletedYData, ...scatterplotDataY])
    setDeletedColorData([...deletedColorData, ...scatterplotDataColor])

    setScatterplotDataX(null)
    setScatterplotDataY(null)
    setScatterplotDataColor(null)
  }

  function handleOptionClick(option) {
    let chosenColor = polygons[contextMenu.polygonID][0].color
    let deletePolygon = false
    let noAction = false  

    switch (option) {
      case 'rbc':
        chosenColor = '#ff0000'
        break
      case 'wbc':
        chosenColor = '#ffffff'   
        break
      case 'plt':
        chosenColor = '#0000ff'
        break
      case 'agg':
        chosenColor = '#00ff00'
        break
      case 'oof':
        chosenColor = '#ffff00'
        break
      case 'delete':
        deletePolygon = true
        let polygonToDelete = [];
        polygonToDelete = polygons.splice(contextMenu.polygonID, 1)


        setDeletedPolygons([...deletedPolygons, ...polygonToDelete])
        setNumberOfDeletedPolygons([...numberOfDeletedPolygons, 1])

        if (scatterplotDataX !== null && scatterplotDataY !== null && scatterplotDataColor !== null) {
          let dataXToDelete = [];
          let dataYToDelete = [];
          let dataColorToDelete = [];


          dataXToDelete = scatterplotDataX.splice(contextMenu.polygonID, 1)
          dataYToDelete = scatterplotDataY.splice(contextMenu.polygonID, 1)
          dataColorToDelete = scatterplotDataColor.splice(contextMenu.polygonID, 1)


          setDeletedXData([...deletedXData, ...dataXToDelete])
          setDeletedYData([...deletedYData, ...dataYToDelete])
          setDeletedColorData([...deletedColorData, ...dataColorToDelete])

          if ((scatterplotDataX.length == 0) || (scatterplotDataY.length === 0) || (scatterplotDataColor.length === 0)) {
              setScatterplotDataX(null)
              setScatterplotDataY(null)
              setScatterplotDataColor(null)
            }
        }
        break
      case 'explain':
        setExplainMenu({ visible: true, polygonID: contextMenu.polygonID })
        break
      case 'noAction':
        noAction = true
    }

    if (!deletePolygon && !noAction) {
      for (let i = 0; i < polygons[contextMenu.polygonID].length; i += 1) {
        polygons[contextMenu.polygonID][i].color = chosenColor
      }
    }

    setContextMenu({ visible: false })
    console.log('Selected', option)
  }

  // Get features names which were used in classification
  function fetchAvailableFeaturesNames() {
    axios
      .get('/available_features_names')
      .then((response) => {
        setAvailableFeaturesNames(response.data.features) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error while getting available features: ${error}`)
      })
  }

  function handleExplainMenuClick(option) {

    // Explain menu handlers
    if (option === 'close') {
      setExplainMenu({ visible: false })
    }
    if (option === 'plot') {
      const randomImageUrl = 'https://www.istockphoto.com/de/foto/wandern-in-den-allg%C3%A4uer-alpen-gm1141196125-305637944';
      setImageUrl(randomImageUrl);

    // Open the image in a new window
      window.open(randomImageUrl, '_blank');
    }
  }

  function handleExplainMenuXAxisSelectorChange(e) {
    console.log(e)
  }

  function handleExplainMenuYAxisSelectorChange(e) {
    console.log(e)
  }

  async function onScatterplotFeatureChangeX(e) {
    const selectedFeature = e.target.value
    setFeatureXAxis(selectedFeature)

    setIsLoading(true);
    axios
      .post('/features_and_data_to_plot', {
        x_feature: selectedFeature, y_feature: featureYAxis
      })
      .then((response) => {
        setScatterplotDataX(response.data.feature_x_values)
        setIsLoading(false);
      })
      .catch((error) => {
        console.error(`Error selecting segmentation method: ${error}`)
      })
  }

  function onScatterplotFeatureChangeY(e) {
    const selectedFeature = e.target.value
    setFeatureYAxis(selectedFeature)

    setIsLoading(true);
    axios
      .post('/features_and_data_to_plot', {
        x_feature: featureXAxis, y_feature: selectedFeature
      })
      .then((response) => {
        setScatterplotDataY(response.data.feature_y_values)
        setIsLoading(false);
      })
      .catch((error) => {
        console.error(`Error selecting segmentation method: ${error}`)
      })
  }

  function onSegmentationMethodChange(e) {
    const selectedMethod = e.target.value

    setIsLoading(true);
    axios
      .post('/select_segmentator', {
        method: selectedMethod,
      })
      .then((response) => {
        setIsLoading(false);
        console.log(response.data) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error selecting segmentation method: ${error}`)
      })
  }

  function onClassificationMethodChange(e) {
    const selectedMethod = e.target.value


    setIsLoading(true);
    axios
      .post('/select_classifier', {
        method: selectedMethod,
      })
      .then((response) => {
        setIsLoading(false);
        console.log(response.data) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error selecting classification method: ${error}`)
      })
  }

  function onDatasetChange(e) {
    const selectedDataset = e.target.value

    console.log(`Selected dataset: ${selectedDataset}`)

    setIsLoading(true);
    axios
      .post('/select_dataset', {
        filename: selectedDataset,
      })
      .then((response) => {
        console.log(response.data) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error selecting dataset: ${error}`)
      })

    setCurrentDataset(selectedDataset)
    setIsLoading(false);
  }


  return (
    <div style={style}>
      <MenuContainer>
        <Menu
          onReset={undoLast}
          onUndo={deleteall}
          onNext={nextImage}
          onPrev={prevImage}
          onSegment={segment}
          onImageId={handleButtonClick}
          onToggleImage={toggleImage}
          onSegmentationMethodChange={onSegmentationMethodChange}
          onClassificationMethodChange={onClassificationMethodChange}
          onDatasetChange={onDatasetChange}
          onClassify={classify}
          onSave={saveMasksAndLabels}
          onDownload={download}
          isClassified={isClassified}
          isSegmented={isSegmented}
          onRetrain={retrain}
          classificationMethods={classificationMethods}
          setClassificationMethods={setClassificationMethods}
          classificationError={classificationError}
          userDataExists={userDataExists}
        />
      </MenuContainer>
      <StageContainer>
        <Stage
          ref={stageRef}
          key='main-stage'
          width={stageDimensions.width}
          height={stageDimensions.height}
          onClick={handleClick}
          onMouseMove={handleMouseMove}
        >
          <Layer key='0'>
            {image && (
              <Image
                width={stageDimensions.width}
                height={stageDimensions.height}
                image={image}
                x={0}
                y={0}
              />
            )}
          </Layer>
          <Layer key='1'>
            {polygons.map((polygon, i) => (
              <Group
                key={`group-${i}-${polygon[0].id}`}
                draggable
                onClick={(e) => {
                  if (e.evt.button === 0) {
                    e.cancelBubble = true
                    console.log('Clicked on polygon', i)
                  }
                }}
                onContextMenu={(e) => {
                  e.evt.preventDefault()
                  const mousePos = stageRef.current
                    .getStage()
                    .getPointerPosition()
                  setContextMenu({
                    visible: true,
                    x: mousePos.x,
                    y: mousePos.y,
                    polygonID: i,
                  })
                  if(mostUncertain && mostUncertain.includes(i)){
                    const index=mostUncertain.indexOf(i)
                    mostUncertain.splice(index,1)}
                }}
                onDragEnd={(e) => {
                  const newPolygon = polygon.map((p) => ({
                    x: p.x + e.target.x(),
                    y: p.y + e.target.y(),
                    color: p.color,
                    id: p.id,
                  }))
                  const newPolygons = [...polygons]
                  newPolygons[i] = newPolygon
                  setPolygons(newPolygons)
                  e.target.position({ x: 0, y: 0 }) // Reset group's position
                }}
                onMouseOver={(e) => {
                  console.log('Mouse over polygon', i)
                }}
              >
                <Line
                  points={polygon.flatMap((p) => [p.x, p.y])}
                  fill={polygon[0].color}
                  opacity={showAmplitudeImage ?0.25 :1}
                  stroke={polygon[0].color}
                  strokeWidth={4}
                  closed
                />

                {polygon.map((point, j) => (
                  <Circle
                    key={`circle-${j}-${point.id}`}
                    x={point.x}
                    y={point.y}
                    radius={3}
                    fill={(() => {
                    // Define your condition here
                    const isHighlighted = i == activePoint; // Replace with your actual condition
                    const isUncertain = mostUncertain && mostUncertain.includes(i);

                    if (isHighlighted) {
                      return 'red'
                    }
                    if (isUncertain) {
                      return '#800080'
                    } else {
                      return '#ffff00'
                    }
                    })()}
                    draggable
                    onDragEnd={(e) => {
                      e.cancelBubble = true // stop event propagation
                      const newPoint = {
                        x: e.target.x() + e.target.getParent().x(),
                        y: e.target.y() + e.target.getParent().y(),
                        id: point.id,
                      }
                      const newPolygon = [...polygon]
                      newPolygon[j] = newPoint
                      const newPolygons = [...polygons]
                      newPolygons[i] = newPolygon
                      setPolygons(newPolygons)
                    }}
                  />
                ))}
              </Group>
            ))}
            {currentPolygon.length > 0 && (
              <Line
                key='preview'
                points={[
                  ...currentPolygon.flatMap((p) => [p.x, p.y]),
                  nextPoint
                    ? nextPoint.x
                    : currentPolygon[currentPolygon.length - 1].x,
                  nextPoint
                    ? nextPoint.y
                    : currentPolygon[currentPolygon.length - 1].y,
                ]}
                stroke='purple'
                strokeWidth={4}
              />
            )}
          </Layer>
        </Stage>
      </StageContainer>
        <Legend>
        </Legend>
      {(availableFeaturesNames.length !== 0) && <Scatterplot 
        featuresList = {availableFeaturesNames}
        featureX={featureXAxis} 
        featureY = {featureYAxis}
        scatterDataX = {scatterplotDataX}
        scatterDataY = {scatterplotDataY}
        scatterDataColor = {scatterplotDataColor}
        onFeatureChangeX={onScatterplotFeatureChangeX}
        onFeatureChangeY={onScatterplotFeatureChangeY} 
        onPointHover={onPointHover}
      />}
      {isLoading && <LoadingSpinner />}
      {contextMenu.visible && (
        <PopupMenu
          x={contextMenu.x}
          y={contextMenu.y}
          handleOptionClick={handleOptionClick}
        />
      )}
      {explainMenu.visible && (
        <ExplainMenu
          handleOptionClick={handleExplainMenuClick}
          selectorOptions={availableFeaturesNames}
          handleXAxisFeatureChange={handleExplainMenuXAxisSelectorChange}
          handleYAxisFeatureChange={handleExplainMenuYAxisSelectorChange}
        />
      )}
    </div>
  )
}

class App extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      counter: 0,
      image: null,
    }
  }

  render() {
    return (
      <div className='App'>
        <div className='App-header'>
          <AnnotationArea />
        </div>
      </div>
    )
  }
}

export default App
