import React, { useEffect, useState } from 'react'
import { Stage, Layer, Line, Image, Circle, Group } from 'react-konva'
import axios from 'axios'
import { Menu, MenuContainer } from './components/Menu/Menu'
import { PopupMenu } from './components/PopupMenu/PopupMenu'
import { ExplainMenu } from './components/ExplainMenu/ExplainMenu'
import { v4 as uuidv4 } from 'uuid'


axios.defaults.baseURL = 'http://localhost:8000'

const stageDimensions = {
  width: 1000,
  height: 800
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
  }

  return <div style={style}>{children}</div>
}

async function getImageWithPredictions(imageId, imageType, callback) {
  // If we show the amplitude image, we want to use it for the masks
  const response = await fetch('http://localhost:8000/images', {
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
  const response = await axios.get('/segment');
  const polygons = response.data.polygons
  callback(polygons)
  return polygons
}



function divideElements(objectOfArrays) {
  const width = stageDimensions.width
  const height = stageDimensions.height  
  const data = {};

  for (let key in objectOfArrays) {
    if (objectOfArrays.hasOwnProperty(key)) {
      data[key] = objectOfArrays[key].map((element) => {
        return {
          x: element.x / width,
          y: element.y / height
        };
      });
    }
  }


  const transformedData = [];

  // Iterate through the original data
  for (const key in data) {
    if (data.hasOwnProperty(key)) {
      const points = data[key];
      const transformedPoints = [];

      // Extract x and y values for each point
      for (const point of points) {
        const { x, y } = point;

        // Create a new object with the desired format
        transformedPoints.push(x, y);
      }

      // Push the new object into the transformed data array
      transformedData.push({ points: transformedPoints });
    }
  }
  return transformedData
}



const AnnotationArea = () => {
  const style = {
    display: 'flex',
    alignItems: 'flex-start', // align items vertically
    justifyContent: 'flex-start', // align items horizontally
    height: '100%', // 100% of the viewport height
    width: '100%', // 100% of the viewport height
    backgroundColor: '#F5F5F5'
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
  const [polygonCounter, setPolygonCounter] = useState(0)
  const [currentPolygon, setCurrentPolygon] = useState([])
  const currentPolygonRef = React.useRef(currentPolygon)
  const [nextPoint, setNextPoint] = useState(null)
  const [deletedPolygons, setDeletedPolygons] = useState([])
  const [numberOfDeletedPolygons, setNumberOfDeletedPolygons] = useState([])
  

  // Preview line management
  const [previewLine, setPreviewLine] = useState(null)
  const isDrawing = React.useRef(false)

  // Context Menu for Polygon-editing
  const [contextMenu, setContextMenu] = useState({ visible: false, x: 0, y: 0, polygonID: -1 });
  const [explainMenu, setExplainMenu] = useState({ visible: false, polygonID: -1 });

  // Component management
  const stageRef = React.useRef()

async function classifyCurrentImage(callback) {
    const masks = divideElements(polygons);

    const response = await fetch('http://localhost:8000/classify', {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ polygons: masks, use_backend_masks: false }),
    })
    const predictions = await response.json()
    console.log(predictions);
    callback(predictions)
    return predictions

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

  function segmentCallback(receivedPolygons) {
    const transformedPolygons = []

    if (receivedPolygons.length !== 0) {
      receivedPolygons.forEach((receivedPolygons, index) => {

        const currentPolygon = []
        for (let i = 0; i < receivedPolygons.points.length; i += 8) {
          currentPolygon.push({
            x: receivedPolygons.points[i] * stageDimensions.width,
            y: receivedPolygons.points[i + 1] * stageDimensions.height,
            color: '#ffa500',
            id: uuidv4()
          })
        }
        transformedPolygons.push(currentPolygon)
      })
    }
    setPolygons(transformedPolygons)

  }

function classifyCallback(labels) {
  const transformedPolygons = polygons.map((polygon, index) => {
    const classId = labels[index]["class_id"];

    console.log(`Polygon ${index + 1} - classId: ${classId}`);

    const color = getColorByClassId(classId);

    return polygon.map((point) => ({
      ...point,
      color: color,
    }));
  });

  setPolygons(transformedPolygons);
}

  function setImageCallback(response_json) {
    // This is a callback function that is called when the image is fetched
    // Its only purpose is to set the image state variables

    setAmplitudeImage(
      `data:image/jpeg;base64,${response_json.amplitude_img_data}`
    )
    setPhaseImage(`data:image/jpeg;base64,${response_json.phase_img_data}`)

    const transformedPolygons = []

    const polygonsWithPredictions = response_json.predictions
    if (polygonsWithPredictions.length !== 0) {
      polygonsWithPredictions.forEach((polygonWithPrediction, index) => {

        const currentPolygon = []
        for (let i = 0; i < polygonWithPrediction.polygon.points.length; i += 8) {
          currentPolygon.push({
            x: polygonWithPrediction.polygon.points[i] * stageDimensions.width,
            y: polygonWithPrediction.polygon.points[i + 1] * stageDimensions.height,
            color: getColorByClassId(polygonWithPrediction.class_id),
            id: uuidv4()
          })
        }
        transformedPolygons.push(currentPolygon)
      })
    }
    setPolygons(transformedPolygons)
  }

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
     setNewImage(imageId, image_type, setImageCallback)
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
      } else if (event.key === 's') {
        saveMask()
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
        { x: mousePos.x, y: mousePos.y, id: uuidv4() },
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
      console.log(polygons)
      setPolygons((prevPolygons) => [...prevPolygons, currentPolygonRef.current])
    }
    setCurrentPolygon([])
    setNextPoint(null)
  }

  const nextImage = () => {
    setImageId((prevId) => prevId + 1)
    setDeletedPolygons([])
    setNumberOfDeletedPolygons([])
  }

  const prevImage = () => {
    setImageId((prevId) => prevId - 1)
    setDeletedPolygons([])
    setNumberOfDeletedPolygons([])
  }

  const handleButtonClick = (e) => {
    e.preventDefault()

    const newImageId = e.target.image_id.value
    // Validate the number
    if (newImageId >= 1 && newImageId <= 1000) {
      setImageId(newImageId)
      // Perform your desired action with the valid number
      console.log('Valid number:', newImageId)
    } else {
      console.log('Invalid number:', newImageId)
    }
  }

  const segment = () => {
    segmentCurrentImage(segmentCallback)
  }

  const classify = () => {
    classifyCurrentImage(classifyCallback)
  }

  const toggleImage = () => {
    setShowAmplitudeImage((prev) => !prev)
  }

  function stopDrawing() {
    saveMask()
    isDrawing.current = false
    setPreviewLine(null)
  }

  function saveMask() {
    console.log('Saving masks...')
    const currentPolygonPoints = currentPolygonRef.current[0].points

    if (currentPolygonPoints.length <= 4) {
      resetCurrentPolygon()
      return
    }

    setPolygonCounter((prevCount) => {
      setPolygons((prevPolygons) => {
        return {
          ...prevPolygons,
          [prevCount]: currentPolygonRef.current.slice(0),
        }
      })
      return prevCount + 1
    })
    setCurrentPolygon([])
  }

  function resetCurrentPolygon() {
    setCurrentPolygon([])
    setPreviewLine(null)
  }

  function undoLast() {    
    let lastNumber = numberOfDeletedPolygons.splice(-1,1)
    lastNumber = lastNumber[0]
    let recoveredPolygon = []

    if ( lastNumber === 1) {
      recoveredPolygon = deletedPolygons.splice(-1,1)            
    } else if ( lastNumber > 1) {
        recoveredPolygon = deletedPolygons.splice(-lastNumber, lastNumber)
    }
    
    polygons.push(...recoveredPolygon)
  }

  function deleteall() {
    setNumberOfDeletedPolygons([...numberOfDeletedPolygons, polygons.length])
    setDeletedPolygons([...deletedPolygons, ...polygons])
    setPolygons([])
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
        let polygonToDelete = []
        polygonToDelete = polygons.splice(contextMenu.polygonID, 1)
        setDeletedPolygons([...deletedPolygons, ...polygonToDelete])
        setNumberOfDeletedPolygons([...numberOfDeletedPolygons, 1])
        break
      case 'explain':
        setExplainMenu({ visible:true, polygonID: contextMenu.polygonID})
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

  function handleExplainMenuClick(option) {
    if (option === 'close') {
      setExplainMenu({ visible: false })
    }
  }

  function onSegmentationMethodChange(e) {
    const selectedMethod = e.target.value

    axios
      .post('/select_segmentator', {
        method: selectedMethod,
      })
      .then((response) => {
        console.log(response.data) // Output the server's response to the console.
      })
      .catch((error) => {
        console.error(`Error selecting segmentation method: ${error}`)
      })
  }

  function onDatasetChange(e) {
    const selectedDataset = e.target.value

    console.log(`Selected dataset: ${selectedDataset}`)

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
          onDatasetChange={onDatasetChange}
          onClassify={classify}
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
                  e.evt.preventDefault();
                  const mousePos = stageRef.current.getStage().getPointerPosition()                  
                  setContextMenu({ visible: true, x:mousePos.x, y:mousePos.y, polygonID: i })
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
                onMouseOver={(e) =>{
                  console.log("Mouse over polygon", i)               
                }}
              >
                <Line
                  points={polygon.flatMap((p) => [p.x, p.y])}
                  fill={polygon[0].color}
                  opacity={0.25}
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
                    fill='#ffff00'
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
                stroke='#000'
                strokeWidth={4}
              />
            )}
            
          </Layer>
        </Stage>
        
      </StageContainer>
      {contextMenu.visible && (<PopupMenu x={contextMenu.x} y={contextMenu.y} handleOptionClick={handleOptionClick}/>)}
      {explainMenu.visible && (<ExplainMenu handleOptionClick={handleExplainMenuClick}/>)}
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
